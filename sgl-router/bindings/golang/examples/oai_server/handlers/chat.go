package handlers

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	sglang "github.com/sglang/sglang-go-grpc-sdk"
	"github.com/valyala/fasthttp"
	"go.uber.org/zap"

	"oai_server/models"
	"oai_server/service"
	"oai_server/utils"
)

// ChatHandler handles chat completion requests
type ChatHandler struct {
	logger  *zap.Logger
	service *service.SGLangService
}

// NewChatHandler creates a new chat handler
func NewChatHandler(logger *zap.Logger, svc *service.SGLangService) *ChatHandler {
	return &ChatHandler{
		logger:  logger,
		service: svc,
	}
}

// HandleChatCompletion handles POST /v1/chat/completions
func (h *ChatHandler) HandleChatCompletion(ctx *fasthttp.RequestCtx) {
	var req models.ChatRequest
	if err := json.Unmarshal(ctx.PostBody(), &req); err != nil {
		h.logger.Warn("Invalid chat completion request", zap.Error(err))
		utils.RespondError(ctx, 400, fmt.Sprintf("Invalid request: %v", err), "invalid_request_error")
		return
	}

	h.logger.Info("Chat completion request received",
		zap.String("model", req.Model),
		zap.Int("messages", len(req.Messages)),
		zap.Bool("stream", req.Stream),
	)

	// Convert to SGLang format
	messages := make([]sglang.ChatMessage, len(req.Messages))
	for i, msg := range req.Messages {
		role, roleOk := msg["role"]
		content, contentOk := msg["content"]

		// Validate role
		if !roleOk || role == "" {
			h.logger.Warn("Missing or empty role in message", zap.Int("message_index", i))
			utils.RespondError(ctx, 400, "Message role is required and cannot be empty", "invalid_request_error")
			return
		}

		// Ensure content is always a string (not null)
		// Chat template requires content field to be present, even if empty
		// If content is missing or null, use empty string
		contentStr := ""
		if contentOk && content != "" {
			contentStr = content
		}

		messages[i] = sglang.ChatMessage{
			Role:    role,
			Content: contentStr, // Always use string, never null
		}
	}

	sglReq := sglang.ChatCompletionRequest{
		Model:    req.Model,
		Messages: messages,
		Stream:   req.Stream,
	}

	if req.Temperature != nil {
		temp := float32(*req.Temperature)
		sglReq.Temperature = &temp
	}
	if req.TopP != nil {
		topP := float32(*req.TopP)
		sglReq.TopP = &topP
	}
	if req.MaxTokens != nil {
		sglReq.MaxCompletionTokens = req.MaxTokens
	}

	// Create context without timeout for streaming requests
	// Note: fasthttp doesn't use standard context.Context, but we create one for the SGLang client
	// Streaming requests should not have a timeout as they can run for a long time
	// The context will be cancelled when the client disconnects or the handler returns
	requestCtx := context.Background()

	if req.Stream {
		h.handleStreamingCompletion(ctx, requestCtx, sglReq)
	} else {
		h.handleNonStreamingCompletion(ctx, requestCtx, sglReq)
	}
}

// isBrokenPipeError checks if the error is a broken pipe error (client disconnected)
func isBrokenPipeError(err error) bool {
	if err == nil {
		return false
	}
	errStr := err.Error()
	return strings.Contains(errStr, "broken pipe") ||
		strings.Contains(errStr, "connection reset by peer") ||
		strings.Contains(errStr, "write: connection closed")
}

func (h *ChatHandler) handleStreamingCompletion(ctx *fasthttp.RequestCtx, requestCtx context.Context, req sglang.ChatCompletionRequest) {
	requestStartTime := time.Now()

	h.logger.Info("Streaming chat completion started", zap.String("model", req.Model))

	// Setup SSE headers
	ctx.SetContentType("text/event-stream")
	ctx.Response.Header.Set("Cache-Control", "no-cache")
	ctx.Response.Header.Set("Connection", "keep-alive")
	ctx.Response.Header.Set("X-Accel-Buffering", "no") // Disable nginx buffering

	// Set status code
	ctx.SetStatusCode(200)

	// CRITICAL: Use SetBodyStreamWriter for true streaming
	// This is the recommended way to implement streaming in fasthttp
	// The callback runs after handler returns, so we need to:
	// 1. Create stream INSIDE the callback to avoid premature cancellation
	// 2. Use independent context (streamCtx) for stream operations
	// 3. Check requestCtx.Done() to detect client disconnection

	// Track metrics (captured in closure)
	var chunkCount int
	var clientDisconnected bool
	var lastChunkTime time.Time
	var chunkIntervals []time.Duration // Track intervals between chunks

	ctx.SetBodyStreamWriter(func(w *bufio.Writer) {
		// CRITICAL: Create independent context for stream operations INSIDE callback
		// This ensures the context is not cancelled when handler returns
		streamCtx, cancel := context.WithCancel(context.Background())
		defer cancel() // Clean up context when callback finishes

		// Create stream INSIDE callback to avoid premature cancellation
		stream, err := h.service.Client().CreateChatCompletionStream(streamCtx, req)
		if err != nil {
			h.logger.Error("Failed to create chat completion stream",
				zap.Error(err),
				zap.String("model", req.Model),
			)
			// Write error response
			w.WriteString("data: {\"error\":{\"message\":\"Failed to create stream\"}}\n\n")
			if flushErr := w.Flush(); flushErr != nil {
				h.logger.Warn("Failed to flush error response", zap.Error(flushErr))
			}
			return
		}
		defer func() {
			if closeErr := stream.Close(); closeErr != nil {
				h.logger.Warn("Failed to close stream", zap.Error(closeErr))
			}
		}()

		defer func() {
			requestEndTime := time.Now()
			totalDuration := requestEndTime.Sub(requestStartTime)

			// Calculate client-side TPOT (average time between sending chunks to client)
			var avgClientTPOT time.Duration
			if len(chunkIntervals) > 0 {
				var sum time.Duration
				for _, interval := range chunkIntervals {
					sum += interval
				}
				avgClientTPOT = sum / time.Duration(len(chunkIntervals))
			}

			if clientDisconnected {
				h.logger.Info("Streaming chat completion interrupted",
					zap.Int("chunks", chunkCount),
					zap.Duration("duration", totalDuration),
					zap.Duration("avg_client_tpot", avgClientTPOT),
				)
			} else {
				h.logger.Info("Streaming chat completion completed",
					zap.Int("chunks", chunkCount),
					zap.Duration("duration", totalDuration),
					zap.Duration("avg_client_tpot", avgClientTPOT),
				)
			}
		}()

		for {
			// Check if original request context is cancelled (client disconnected)
			select {
			case <-requestCtx.Done():
				clientDisconnected = true
				return
			case <-streamCtx.Done():
				return
			default:
			}

			// OPTIMIZATION: Use RecvJSON() to get raw JSON string directly from Rust FFI
			// This avoids JSON parsing and re-serialization overhead, matching Rust performance
			// Rust FFI already returns complete OpenAI-format JSON, we just need to add SSE prefix/suffix
			chunkJSON, err := stream.RecvJSON()

			if err == io.EOF {
				break
			}
			if err != nil {
				h.logger.Error("Stream error",
					zap.Error(err),
					zap.Int("chunks_sent", chunkCount),
				)
				break
			}
			if chunkJSON == "" {
				continue
			}
			chunkCount++

			// Track TPOT: record time when we send each chunk to client
			currentChunkTime := time.Now()
			if chunkCount > 1 {
				// Calculate interval since last chunk
				chunkIntervals = append(chunkIntervals, currentChunkTime.Sub(lastChunkTime))
			}
			lastChunkTime = currentChunkTime

			// OPTIMIZATION: Directly use Rust FFI JSON string - no parsing/serialization needed!
			// Just add SSE prefix and suffix
			w.WriteString("data: ")
			w.WriteString(chunkJSON)
			w.WriteString("\n\n")

			// CRITICAL: Flush immediately after each chunk
			// This ensures data is sent immediately, reducing TTFT
			if err := w.Flush(); err != nil {
				if isBrokenPipeError(err) {
					clientDisconnected = true
					return
				}
				h.logger.Warn("Flush error", zap.Error(err))
			}
		}

		// Send done message
		if !clientDisconnected {
			w.WriteString("data: [DONE]\n\n")
			if err := w.Flush(); err != nil {
				if !isBrokenPipeError(err) {
					h.logger.Warn("Final flush error", zap.Error(err))
				}
			}
		}
	})
}

func (h *ChatHandler) handleNonStreamingCompletion(ctx *fasthttp.RequestCtx, requestCtx context.Context, req sglang.ChatCompletionRequest) {
	startTime := time.Now()
	resp, err := h.service.Client().CreateChatCompletion(requestCtx, req)
	if err != nil {
		h.logger.Error("Failed to create chat completion",
			zap.Error(err),
			zap.String("model", req.Model),
		)
		utils.RespondError(ctx, 500, fmt.Sprintf("Failed to create completion: %v", err), "server_error")
		return
	}

	duration := time.Since(startTime)
	h.logger.Info("Chat completion completed",
		zap.String("model", req.Model),
		zap.Duration("duration", duration),
		zap.Int("choices", len(resp.Choices)),
	)

	// Convert to OpenAI format
	response := utils.BuildResponseBase(resp.ID, resp.Created, resp.Model)
	response["object"] = "chat.completion"

	choices := make([]map[string]interface{}, len(resp.Choices))
	for i, choice := range resp.Choices {
		choiceMap := map[string]interface{}{
			"index": choice.Index,
			"message": map[string]interface{}{
				"role":    choice.Message.Role,
				"content": choice.Message.Content,
			},
			"finish_reason": choice.FinishReason,
		}
		if len(choice.Message.ToolCalls) > 0 {
			toolCalls := make([]map[string]interface{}, len(choice.Message.ToolCalls))
			for j, tc := range choice.Message.ToolCalls {
				toolCalls[j] = map[string]interface{}{
					"id":       tc.ID,
					"type":     tc.Type,
					"function": map[string]interface{}{"name": tc.Function.Name, "arguments": tc.Function.Arguments},
				}
			}
			choiceMap["message"].(map[string]interface{})["tool_calls"] = toolCalls
		}
		choices[i] = choiceMap
	}
	response["choices"] = choices

	// Usage is always present (not a pointer)
	response["usage"] = map[string]interface{}{
		"prompt_tokens":     resp.Usage.PromptTokens,
		"completion_tokens": resp.Usage.CompletionTokens,
		"total_tokens":      resp.Usage.TotalTokens,
	}

	ctx.SetStatusCode(200)
	ctx.SetContentType("application/json")
	jsonData, _ := json.Marshal(response)
	ctx.Write(jsonData)
}
