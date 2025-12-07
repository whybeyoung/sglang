package handlers

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

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

	path := string(ctx.Path())

	defer func() {
		statusCode := ctx.Response.StatusCode()
		if statusCode == 0 {
			statusCode = 200
		}
		h.logHTTPResponse(statusCode, path)
	}()

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
			Content: contentStr,
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
	if req.MaxCompletionTokens != nil {
		sglReq.MaxCompletionTokens = req.MaxCompletionTokens
	} else if req.MaxTokens != nil {
		sglReq.MaxCompletionTokens = req.MaxTokens
	}

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
		strings.Contains(errStr, "connection closed") ||
		strings.Contains(errStr, "write: connection closed")
}

// logHTTPResponse logs HTTP response with colored output
func (h *ChatHandler) logHTTPResponse(statusCode int, path string) {
	var statusText string
	var colorCode string

	switch {
	case statusCode >= 200 && statusCode < 300:
		colorCode = "\033[32m" // Green
		statusText = "OK"
	case statusCode >= 300 && statusCode < 400:
		colorCode = "\033[33m" // Yellow
		statusText = "Redirect"
	case statusCode >= 400 && statusCode < 500:
		colorCode = "\033[33m" // Yellow
		statusText = "Client Error"
	case statusCode >= 500:
		colorCode = "\033[31m" // Red
		statusText = "Server Error"
	default:
		colorCode = "\033[37m" // White
		statusText = "Unknown"
	}

	resetCode := "\033[0m"
	msg := fmt.Sprintf("%s[%d %s]%s %s", colorCode, statusCode, statusText, resetCode, path)
	h.logger.Info(msg)
}

func (h *ChatHandler) handleStreamingCompletion(ctx *fasthttp.RequestCtx, requestCtx context.Context, req sglang.ChatCompletionRequest) {

	ctx.SetContentType("text/event-stream")
	ctx.Response.Header.Set("Cache-Control", "no-cache")
	ctx.Response.Header.Set("Connection", "keep-alive")
	ctx.Response.Header.Set("X-Accel-Buffering", "no")
	ctx.SetStatusCode(200)

	var clientDisconnected bool

	ctx.SetBodyStreamWriter(func(w *bufio.Writer) {
		streamCtx, cancel := context.WithCancel(context.Background())
		defer cancel()

		stream, err := h.service.Client().CreateChatCompletionStream(streamCtx, req)
		if err != nil {
			h.logger.Error("Failed to create chat completion stream",
				zap.Error(err),
				zap.String("model", req.Model),
			)
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

		for {
			if clientDisconnected {
				cancel()
				return
			}

			var chunkJSON string
			var err error

			recvDone := make(chan struct{})
			go func() {
				defer close(recvDone)
				chunkJSON, err = stream.RecvJSON()
			}()

			select {
			case <-streamCtx.Done():
				return
			case <-recvDone:
				if err == io.EOF {
					if !clientDisconnected {
						w.WriteString("data: [DONE]\n\n")
						if flushErr := w.Flush(); flushErr != nil {
							if !isBrokenPipeError(flushErr) {
								h.logger.Warn("Final flush error", zap.Error(flushErr))
							}
						}
					}
					return
				}
				if err != nil {
					if err == context.Canceled || err == context.DeadlineExceeded {
						return
					}
					h.logger.Error("Stream error", zap.Error(err))
					return
				}
				if chunkJSON == "" {
					continue
				}

				w.WriteString("data: ")
				w.WriteString(chunkJSON)
				w.WriteString("\n\n")

				if err := w.Flush(); err != nil {
					if isBrokenPipeError(err) {
						clientDisconnected = true
						cancel()
						return
					}
					h.logger.Warn("Flush error", zap.Error(err))
				}
			}
		}
	})
}

func (h *ChatHandler) handleNonStreamingCompletion(ctx *fasthttp.RequestCtx, requestCtx context.Context, req sglang.ChatCompletionRequest) {
	resp, err := h.service.Client().CreateChatCompletion(requestCtx, req)
	if err != nil {
		h.logger.Error("Failed to create chat completion",
			zap.Error(err),
			zap.String("model", req.Model),
		)
		utils.RespondError(ctx, 500, fmt.Sprintf("Failed to create completion: %v", err), "server_error")
		return
	}

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
