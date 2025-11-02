package stages

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/sglang/sglang-router-go/internal/router/pipeline"
	"github.com/sglang/sglang-router-go/internal/tokenizer"
	"github.com/sglang/sglang-router-go/pkg/proto"
	"go.uber.org/zap"
)

// StreamProcessor handles streaming response processing
// Similar to Rust streaming::StreamingProcessor
type StreamProcessor struct {
	logger *zap.Logger
	// tokenizer interface{} // TODO: Add tokenizer
}

// NewStreamProcessor creates a new stream processor
func NewStreamProcessor(logger *zap.Logger) *StreamProcessor {
	return &StreamProcessor{
		logger: logger,
	}
}

// ProcessStreamingResponse processes a streaming response and returns SSE response
// Similar to Rust StreamingProcessor::process_streaming_response
func (sp *StreamProcessor) ProcessStreamingResponse(
	ctx context.Context,
	execResult *pipeline.ExecutionResult,
	dispatch *pipeline.DispatchMetadata,
	isChat bool,
	components *pipeline.SharedComponents,
) (interface{}, error) {
	// Create a pipe for SSE streaming
	// Similar to Rust: mpsc::unbounded_channel
	reader, writer := io.Pipe()

	// Start background goroutine to process stream
	// Similar to Rust: tokio::spawn
	go func() {
		defer writer.Close()

		var err error
		if execResult.IsDual {
			// Dual mode (PD): process both prefill and decode streams
			// TODO: Implement dual stream processing
			sp.logger.Warn("Dual stream processing not yet fully implemented")
			err = fmt.Errorf("dual stream processing not yet implemented")
		} else {
			// Single mode: process single stream
			stream, ok := execResult.Single.(proto.SglangScheduler_GenerateClient)
			if !ok {
				err = fmt.Errorf("invalid stream type: expected proto.SglangScheduler_GenerateClient, got %T", execResult.Single)
			} else {
				if isChat {
					err = sp.processStreamingChat(ctx, stream, dispatch, writer, components)
				} else {
					err = sp.processStreamingGenerate(ctx, stream, dispatch, writer, components)
				}
			}
		}

		// Send error if any
		if err != nil {
			errorEvent := map[string]interface{}{
				"error": map[string]interface{}{
					"message": err.Error(),
					"type":    "internal_error",
				},
			}
			if jsonBytes, marshalErr := json.Marshal(errorEvent); marshalErr == nil {
				fmt.Fprintf(writer, "data: %s\n\n", string(jsonBytes))
			}
		}

		// Send [DONE] event
		fmt.Fprintf(writer, "data: [DONE]\n\n")
	}()

	// Return StreamingResponse
	// Create StreamingResponse without importing router package (to avoid cycle)
	streamingResp := &pipeline.StreamingResponse{
		Reader:      reader,
		ContentType: "text/event-stream",
		Headers: map[string]string{
			"Cache-Control": "no-cache",
			"Connection":    "keep-alive",
		},
	}
	return streamingResp, nil
}

// ProcessStreamingGenerate processes a streaming generate response
func (sp *StreamProcessor) ProcessStreamingGenerate(
	ctx context.Context,
	execResult *pipeline.ExecutionResult,
	dispatch *pipeline.DispatchMetadata,
	components *pipeline.SharedComponents,
) (interface{}, error) {
	return sp.ProcessStreamingResponse(ctx, execResult, dispatch, false, components)
}

// ProcessStreamingChat processes a streaming chat response
func (sp *StreamProcessor) ProcessStreamingChat(
	ctx context.Context,
	execResult *pipeline.ExecutionResult,
	dispatch *pipeline.DispatchMetadata,
	components *pipeline.SharedComponents,
) (interface{}, error) {
	return sp.ProcessStreamingResponse(ctx, execResult, dispatch, true, components)
}

// processStreamingChat processes a streaming chat completion response
// Similar to Rust process_streaming_chunks for chat
// NOTE: Full implementation requires tokenizer for detokenization
func (sp *StreamProcessor) processStreamingChat(
	ctx context.Context,
	stream proto.SglangScheduler_GenerateClient,
	dispatch *pipeline.DispatchMetadata,
	writer io.Writer,
	components *pipeline.SharedComponents,
) error {
	// Track state per index (for n>1 case)
	streamBuffers := make(map[uint32]string)
	finishReasons := make(map[uint32]string)
	isFirst := make(map[uint32]bool)
	// TODO: Extract stop params from request and create stop decoder
	// For now, use empty stop sequences
	// skipSpecialTokens will be used when tokenizer is integrated

	// Main streaming loop
	for {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Receive next response from stream
		genResponse, err := stream.Recv()
		if err == io.EOF {
			// Stream ended
			break
		}
		if err != nil {
			return fmt.Errorf("stream receive error: %w", err)
		}

		// Process response based on type
		if chunk := genResponse.GetChunk(); chunk != nil {
			index := chunk.Index

			// Initialize state for this index if first time
			if !isFirst[index] {
				isFirst[index] = true
				streamBuffers[index] = ""
			}

			// Decode tokens to text using tokenizer
			tokenIDs := chunk.TokenIds
			if len(tokenIDs) == 0 {
				continue
			}

			// Decode tokens using tokenizer if available
			chunkText := ""
			if components != nil && components.Tokenizer != nil {
				if tok, ok := components.Tokenizer.(tokenizer.Tokenizer); ok {
					text, err := tok.Decode(tokenIDs, true) // skipSpecialTokens=true
					if err != nil {
						sp.logger.Warn("Tokenizer decode failed in streaming",
							zap.Error(err),
							zap.Uint32("index", index),
							zap.Int("token_count", len(tokenIDs)),
						)
						// Fallback to placeholder
						chunkText = fmt.Sprintf("[tokens: %d]", len(tokenIDs))
					} else {
						chunkText = text
						sp.logger.Debug("Tokens decoded in streaming",
							zap.Uint32("index", index),
							zap.String("text_preview", text[:min(len(text), 50)]),
						)
					}
				}
			}

			// Fallback if tokenizer not available
			if chunkText == "" {
				chunkText = fmt.Sprintf("[tokens: %d]", len(tokenIDs))
				sp.logger.Debug("Using placeholder token decode",
					zap.Uint32("index", index),
					zap.Int("token_count", len(tokenIDs)),
				)
			}

			// TODO: Process through stop decoder for proper stop sequence handling
			// For now, use direct tokenizer decode
			// In Rust: Self::process_chunk_tokens(stop_decoder, &chunk.token_ids)
			// When stop decoder is available:
			//   stopDecoder := getOrCreateStopDecoder(index, components, ctx.Input.Request)
			//   chunkText, shouldStop := ProcessChunkTokens(stopDecoder, tokenIDs)
			//   if shouldStop { break }

			// Accumulate text
			streamBuffers[index] += chunkText

			// Build SSE event for OpenAI-compatible chat completion
			// Similar to Rust: build chat completion SSE chunk
			choice := map[string]interface{}{
				"index": index,
				"delta": map[string]interface{}{
					"role":    "assistant",
					"content": chunkText,
				},
				"finish_reason": nil,
			}

			event := map[string]interface{}{
				"id":      fmt.Sprintf("%s-%d", dispatch.RequestID, index),
				"object":  "chat.completion.chunk",
				"created": dispatch.Created,
				"model":   dispatch.Model,
				"choices": []interface{}{choice},
			}

			// Format as SSE and write
			if jsonBytes, marshalErr := json.Marshal(event); marshalErr == nil {
				fmt.Fprintf(writer, "data: %s\n\n", string(jsonBytes))
				sp.logger.Debug("Streaming chunk sent",
					zap.Uint32("index", index),
					zap.String("text_preview", chunkText),
				)
			}

		} else if complete := genResponse.GetComplete(); complete != nil {
			index := complete.Index

			// Final chunk for this index
			finishReason := complete.FinishReason
			finishReasons[index] = finishReason

			// Build final SSE event
			choice := map[string]interface{}{
				"index":         index,
				"delta":         map[string]interface{}{},
				"finish_reason": finishReason,
			}

			event := map[string]interface{}{
				"id":      fmt.Sprintf("%s-%d", dispatch.RequestID, index),
				"object":  "chat.completion.chunk",
				"created": dispatch.Created,
				"model":   dispatch.Model,
				"choices": []interface{}{choice},
			}

			if jsonBytes, marshalErr := json.Marshal(event); marshalErr == nil {
				fmt.Fprintf(writer, "data: %s\n\n", string(jsonBytes))
				sp.logger.Debug("Streaming complete sent",
					zap.Uint32("index", index),
					zap.String("finish_reason", finishReason),
				)
			}

		} else if errResp := genResponse.GetError(); errResp != nil {
			return fmt.Errorf("stream error: %s", errResp.Message)
		}
	}

	return nil
}

// processStreamingGenerate processes a streaming generate response
// Similar to Rust process_generate_streaming
// NOTE: Full implementation requires tokenizer for detokenization
func (sp *StreamProcessor) processStreamingGenerate(
	ctx context.Context,
	stream proto.SglangScheduler_GenerateClient,
	dispatch *pipeline.DispatchMetadata,
	writer io.Writer,
	components *pipeline.SharedComponents,
) error {
	// Track state per index (for n>1 case)
	accumulatedTexts := make(map[uint32]string)
	completionTokensMap := make(map[uint32]uint32)

	// Main streaming loop
	for {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Receive next response from stream
		genResponse, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("stream receive error: %w", err)
		}

		// Process response
		if chunk := genResponse.GetChunk(); chunk != nil {
			index := chunk.Index

			// Update completion tokens
			completionTokens := completionTokensMap[index]
			completionTokens += uint32(len(chunk.TokenIds))
			completionTokensMap[index] = completionTokens

			// Decode tokens using tokenizer
			tokenIDs := chunk.TokenIds
			chunkText := ""
			if components != nil && components.Tokenizer != nil {
				if tok, ok := components.Tokenizer.(tokenizer.Tokenizer); ok {
					text, err := tok.Decode(tokenIDs, true) // skipSpecialTokens=true to handle newlines correctly
					if err != nil {
						sp.logger.Warn("Tokenizer decode failed in generate streaming",
							zap.Error(err),
							zap.Uint32("index", index),
							zap.Int("token_count", len(tokenIDs)),
						)
						chunkText = fmt.Sprintf("[tokens: %d]", len(tokenIDs))
					} else {
						chunkText = text
						sp.logger.Debug("Tokens decoded in generate streaming",
							zap.Uint32("index", index),
						)
					}
				}
			}

			// Fallback if tokenizer not available
			if chunkText == "" {
				chunkText = fmt.Sprintf("[tokens: %d]", len(tokenIDs))
			}

			// Accumulate text
			accumulatedText := accumulatedTexts[index]
			accumulatedText += chunkText
			accumulatedTexts[index] = accumulatedText

			// Build SGLang format streaming response
			chunkResponse := map[string]interface{}{
				"text": chunkText,
				// TODO: Add logprobs if requested
			}

			// Format as SSE and write (SGLang format)
			if jsonBytes, marshalErr := json.Marshal(chunkResponse); marshalErr == nil {
				fmt.Fprintf(writer, "data: %s\n\n", string(jsonBytes))
				sp.logger.Debug("Generate streaming chunk sent",
					zap.Uint32("index", index),
					zap.String("accumulated_preview", accumulatedText),
				)
			}

		} else if complete := genResponse.GetComplete(); complete != nil {
			index := complete.Index

			// Final response for this index
			accumulatedText := accumulatedTexts[index]
			completionTokens := completionTokensMap[index]

			finishResponse := map[string]interface{}{
				"text":              accumulatedText,
				"finish_reason":     complete.FinishReason,
				"prompt_tokens":     complete.PromptTokens,
				"completion_tokens": completionTokens,
				"cached_tokens":     complete.CachedTokens,
			}

			// Format as SSE and write
			if jsonBytes, marshalErr := json.Marshal(finishResponse); marshalErr == nil {
				fmt.Fprintf(writer, "data: %s\n\n", string(jsonBytes))
				sp.logger.Debug("Generate streaming complete sent",
					zap.Uint32("index", index),
					zap.String("finish_reason", complete.FinishReason),
				)
			}

		} else if errResp := genResponse.GetError(); errResp != nil {
			return fmt.Errorf("stream error: %s", errResp.Message)
		}
	}

	return nil
}
