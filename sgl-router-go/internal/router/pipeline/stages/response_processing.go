package stages

import (
	"encoding/json"
	"fmt"

	"github.com/sglang/sglang-router-go/internal/protocols"
	"github.com/sglang/sglang-router-go/internal/router/pipeline"
	"go.uber.org/zap"
)

// ResponseProcessingStage processes and formats responses
// Similar to Rust ResponseProcessingStage
// NOTE: Full implementation requires:
// 1. Stream processing (proto stream reading)
// 2. Stop sequence decoding
// 3. Tokenizer for detokenization
// 4. Tool call parsing (if applicable)
// 5. Response formatting
type ResponseProcessingStage struct {
	*pipeline.BaseStage
	streamProcessor *StreamProcessor
}

// NewResponseProcessingStage creates a new response processing stage
func NewResponseProcessingStage(logger *zap.Logger) *ResponseProcessingStage {
	return &ResponseProcessingStage{
		BaseStage:       pipeline.NewBaseStage("ResponseProcessing", logger),
		streamProcessor: NewStreamProcessor(logger),
	}
}

// Execute implements PipelineStage
func (s *ResponseProcessingStage) Execute(ctx *pipeline.RequestContext) (interface{}, error) {
	execResult := ctx.State.Response.ExecutionResult
	if execResult == nil {
		return nil, fmt.Errorf("request execution stage not completed")
	}

	prep := ctx.State.Preparation
	if prep == nil {
		return nil, fmt.Errorf("preparation stage not completed")
	}

	dispatch := ctx.State.Dispatch
	if dispatch == nil {
		return nil, fmt.Errorf("dispatch metadata stage not completed")
	}

	// Determine if streaming
	isStreaming := dispatch.IsStreaming

	if isStreaming {
		// Process streaming response
		// Similar to Rust: streaming_processor.process_streaming_response()
		// This should return an SSE HTTP response (early pipeline exit)
		var streamResp interface{}
		var err error

		requestCtx := ctx.Context() // Get actual context from method

		if ctx.Input.RequestType == pipeline.RequestTypeChat {
			streamResp, err = s.streamProcessor.ProcessStreamingChat(
				requestCtx,
				execResult,
				dispatch,
			)
		} else {
			streamResp, err = s.streamProcessor.ProcessStreamingGenerate(
				requestCtx,
				execResult,
				dispatch,
			)
		}

		if err != nil {
			return nil, fmt.Errorf("streaming processing failed: %w", err)
		}

		// Store streaming response (for HTTP server to return)
		ctx.State.Response.StreamingResponse = streamResp

		s.Logger.Debug("Streaming response prepared",
			zap.String("request_id", dispatch.RequestID),
		)

		// Return early (similar to Rust Ok(Some(response)))
		return streamResp, nil
	}

	// Non-streaming response processing
	// TODO: Process non-streaming response
	// In Rust: processor.process_non_streaming_chat_response() or process_non_streaming_generate_response()
	// This involves:
	// 1. Reading all chunks from stream
	// 2. Detokenizing tokens
	// 3. Applying stop sequence trimming
	// 4. Formatting as ChatCompletionResponse or GenerateResponse

	// For now, create a placeholder response
	var finalResponse interface{}

	if ctx.Input.RequestType == pipeline.RequestTypeChat {
		finalResponse = &protocols.ChatCompletionResponse{
			ID:      dispatch.RequestID,
			Object:  "chat.completion",
			Created: dispatch.Created,
			Model:   dispatch.Model,
			Choices: []protocols.ChatCompletionChoice{
				{
					Index:        0,
					Message:      protocols.ChatMessage{Role: "assistant", Content: json.RawMessage(`"Placeholder response - tokenizer and stream processing required"`)},
					FinishReason: "length",
				},
			},
			Usage: &protocols.Usage{
				PromptTokens:     0,
				CompletionTokens: 0,
				TotalTokens:      0,
			},
		}
	} else {
		finalResponse = &protocols.GenerateResponse{
			RequestID:        dispatch.RequestID,
			Text:             "Placeholder response - tokenizer and stream processing required",
			FinishReason:     "length",
			PromptTokens:     0,
			CompletionTokens: 0,
		}
	}

	ctx.State.Response.FinalResponse = finalResponse

	s.Logger.Debug("Response processed (placeholder)",
		zap.String("request_id", dispatch.RequestID),
		zap.Bool("is_streaming", isStreaming),
	)

	return nil, nil // Pipeline complete
}
