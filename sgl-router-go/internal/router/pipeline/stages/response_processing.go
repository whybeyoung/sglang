package stages

import (
	"fmt"

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
	streamProcessor   *StreamProcessor
	responseProcessor *ResponseProcessor
}

// NewResponseProcessingStage creates a new response processing stage
func NewResponseProcessingStage(logger *zap.Logger) *ResponseProcessingStage {
	return &ResponseProcessingStage{
		BaseStage:         pipeline.NewBaseStage("ResponseProcessing", logger),
		streamProcessor:   NewStreamProcessor(logger),
		responseProcessor: NewResponseProcessor(logger),
	}
}

// Execute implements PipelineStage
// Similar to Rust ResponseProcessingStage::execute
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
	// Similar to Rust: processor.process_non_streaming_chat_response() or process_non_streaming_generate_response()
	var finalResponse interface{}

	// Get stop decoder from context (created in PreparationStage)
	stopDecoder := ctx.State.Response.StopDecoder
	// TODO: Extract request_logprobs from request
	requestLogprobs := false

	if ctx.Input.RequestType == pipeline.RequestTypeChat {
		// Process chat response
		chatResponse, procErr := s.responseProcessor.ProcessNonStreamingChatResponse(
			execResult,
			dispatch,
			stopDecoder,
			requestLogprobs,
		)
		if procErr != nil {
			return nil, fmt.Errorf("failed to process chat response: %w", procErr)
		}
		finalResponse = chatResponse
	} else {
		// Process generate response
		generateResponse, procErr := s.responseProcessor.ProcessNonStreamingGenerateResponse(
			execResult,
			dispatch,
			stopDecoder,
			requestLogprobs,
		)
		if procErr != nil {
			return nil, fmt.Errorf("failed to process generate response: %w", procErr)
		}
		finalResponse = generateResponse
	}

	ctx.State.Response.FinalResponse = finalResponse

	s.Logger.Debug("Response processed",
		zap.String("request_id", dispatch.RequestID),
		zap.Bool("is_streaming", isStreaming),
	)

	return nil, nil // Pipeline complete
}
