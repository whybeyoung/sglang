package stages

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/sglang/sglang-router-go/internal/protocols"
	"github.com/sglang/sglang-router-go/internal/router/pipeline"
	"github.com/sglang/sglang-router-go/internal/tokenizer"
	"github.com/sglang/sglang-router-go/pkg/proto"
	"go.uber.org/zap"
)

// Note: tokenizer package is imported for StopSequenceDecoder interface

// ResponseProcessor handles non-streaming response processing
// Similar to Rust processing::ResponseProcessor
// NOTE: Full implementation requires:
// 1. Stream collection (proto stream reading)
// 2. Tokenizer for detokenization
// 3. Stop sequence decoding
// 4. Tool call parsing (if applicable)
type ResponseProcessor struct {
	logger *zap.Logger
	// tokenizer interface{} // TODO: Add tokenizer
	// toolParserFactory interface{} // TODO: Add tool parser factory
	// reasoningParserFactory interface{} // TODO: Add reasoning parser factory
}

// NewResponseProcessor creates a new response processor
func NewResponseProcessor(logger *zap.Logger) *ResponseProcessor {
	return &ResponseProcessor{
		logger: logger,
	}
}

// ProcessNonStreamingChatResponse processes a non-streaming chat response
// Similar to Rust process_non_streaming_chat_response
func (rp *ResponseProcessor) ProcessNonStreamingChatResponse(
	execResult *pipeline.ExecutionResult,
	dispatch *pipeline.DispatchMetadata,
	stopDecoder interface{}, // TODO: Use proper StopDecoder type
	requestLogprobs bool,
	components *pipeline.SharedComponents,
) (*protocols.ChatCompletionResponse, error) {
	// Step 1: Collect all responses from stream
	// Similar to Rust collect_and_merge_responses
	allResponses, err := rp.collectAndMergeResponses(execResult, requestLogprobs)
	if err != nil {
		return nil, fmt.Errorf("failed to collect responses: %w", err)
	}

	if len(allResponses) == 0 {
		return nil, fmt.Errorf("no responses from server")
	}

	// Step 2: Process each choice
	// Similar to Rust process_single_choice
	var choices []protocols.ChatCompletionChoice
	for idx, complete := range allResponses {
		choice, err := rp.processSingleChoice(
			complete,
			idx,
			stopDecoder,
			components,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to process choice %d: %w", idx, err)
		}
		choices = append(choices, *choice)
	}

	// Step 3: Build usage statistics
	var totalPromptTokens, totalCompletionTokens uint32
	for _, resp := range allResponses {
		totalPromptTokens += uint32(resp.GetPromptTokens())
		totalCompletionTokens += uint32(resp.GetCompletionTokens())
	}

	usage := &protocols.Usage{
		PromptTokens:     int(totalPromptTokens),
		CompletionTokens: int(totalCompletionTokens),
		TotalTokens:      int(totalPromptTokens + totalCompletionTokens),
	}

	// Step 4: Build final response
	response := &protocols.ChatCompletionResponse{
		ID:      dispatch.RequestID,
		Object:  "chat.completion",
		Created: dispatch.Created,
		Model:   dispatch.Model,
		Choices: choices,
		Usage:   usage,
	}

	return response, nil
}

// ProcessNonStreamingGenerateResponse processes a non-streaming generate response
// Similar to Rust process_non_streaming_generate_response
func (rp *ResponseProcessor) ProcessNonStreamingGenerateResponse(
	execResult *pipeline.ExecutionResult,
	dispatch *pipeline.DispatchMetadata,
	stopDecoder interface{}, // TODO: Use proper StopDecoder type
	requestLogprobs bool,
	components *pipeline.SharedComponents,
) (*protocols.GenerateResponse, error) {
	// Step 1: Collect all responses from stream
	allResponses, err := rp.collectAndMergeResponses(execResult, requestLogprobs)
	if err != nil {
		return nil, fmt.Errorf("failed to collect responses: %w", err)
	}

	if len(allResponses) == 0 {
		return nil, fmt.Errorf("no responses from server")
	}

	// For generate, use the first response (n=1 by default)
	complete := allResponses[0]

	// Step 2: Process tokens through stop decoder
	// TODO: Use actual stop decoder
	decodedText, err := rp.decodeTokens(complete.OutputIds, stopDecoder, components)
	if err != nil {
		return nil, fmt.Errorf("failed to decode tokens: %w", err)
	}

	// Step 3: Build response
	response := &protocols.GenerateResponse{
		RequestID:        dispatch.RequestID,
		Text:             decodedText,
		FinishReason:     complete.FinishReason,
		PromptTokens:     int(complete.GetPromptTokens()),
		CompletionTokens: int(complete.GetCompletionTokens()),
	}

	return response, nil
}

// collectAndMergeResponses collects responses from execution result
// Similar to Rust collect_and_merge_responses
func (rp *ResponseProcessor) collectAndMergeResponses(
	execResult *pipeline.ExecutionResult,
	requestLogprobs bool,
) ([]*proto.GenerateComplete, error) {
	// Create a context for collection (use background context since we're already in a pipeline stage)
	ctx := context.Background()

	if execResult.IsDual {
		// PD mode: collect from both prefill and decode
		prefillStream, ok1 := execResult.Dual.Prefill.(proto.SglangScheduler_GenerateClient)
		decodeStream, ok2 := execResult.Dual.Decode.(proto.SglangScheduler_GenerateClient)
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("invalid stream types in dual mode: prefill=%T, decode=%T",
				execResult.Dual.Prefill, execResult.Dual.Decode)
		}

		// Collect prefill for input_logprobs (don't mark completed yet)
		prefillResponses, err := collectStreamChunks(ctx, prefillStream, rp.logger)
		if err != nil {
			return nil, fmt.Errorf("failed to collect prefill responses: %w", err)
		}

		// Collect decode for actual output
		decodeResponses, err := collectStreamChunks(ctx, decodeStream, rp.logger)
		if err != nil {
			return nil, fmt.Errorf("failed to collect decode responses: %w", err)
		}

		// Merge input_logprobs from prefill if requested
		if requestLogprobs {
			if len(prefillResponses) > 0 {
				if prefillInputLogprobs := prefillResponses[0].InputLogprobs; prefillInputLogprobs != nil {
					// Copy input_logprobs to decode responses
					for _, decodeResp := range decodeResponses {
						// Note: proto messages are immutable, so we'd need to create a new message
						// For now, just log - full implementation would require proto message manipulation
						_ = prefillInputLogprobs
						rp.logger.Debug("Input logprobs available from prefill",
							zap.Uint32("decode_index", decodeResp.Index),
						)
					}
				}
			}
		}

		// Return decode responses (these contain the actual output)
		if len(decodeResponses) == 0 {
			return nil, fmt.Errorf("no decode responses collected")
		}
		return decodeResponses, nil
	}

	// Single mode: collect from single stream
	stream, ok := execResult.Single.(proto.SglangScheduler_GenerateClient)
	if !ok {
		return nil, fmt.Errorf("invalid stream type: expected proto.SglangScheduler_GenerateClient, got %T", execResult.Single)
	}

	responses, err := collectStreamChunks(ctx, stream, rp.logger)
	if err != nil {
		return nil, fmt.Errorf("failed to collect stream responses: %w", err)
	}

	return responses, nil
}

// processSingleChoice processes a single choice from complete response
// Similar to Rust process_single_choice
func (rp *ResponseProcessor) processSingleChoice(
	complete *proto.GenerateComplete,
	index int,
	stopDecoder interface{},
	components *pipeline.SharedComponents,
) (*protocols.ChatCompletionChoice, error) {
	// Step 1: Decode tokens
	// TODO: Use actual stop decoder
	decodedText, err := rp.decodeTokens(complete.OutputIds, stopDecoder, components)
	if err != nil {
		return nil, fmt.Errorf("failed to decode tokens: %w", err)
	}

	// Step 2: Build message
	contentBytes, _ := json.Marshal(decodedText)
	message := protocols.ChatMessage{
		Role:    "assistant",
		Content: json.RawMessage(contentBytes),
	}

	// Step 3: Build choice
	choice := &protocols.ChatCompletionChoice{
		Index:        index,
		Message:      message,
		FinishReason: complete.FinishReason,
	}

	return choice, nil
}

// decodeTokens decodes token IDs to text
// Similar to Rust process_single_choice token decoding logic
// TODO: Implement with actual tokenizer and stop decoder
func (rp *ResponseProcessor) decodeTokens(
	tokenIDs []uint32,
	stopDecoder interface{},
	components *pipeline.SharedComponents,
) (string, error) {
	if len(tokenIDs) == 0 {
		return "", nil
	}

	// Step 1: Try to use tokenizer directly if available
	// Note: Full implementation should use stop decoder to process tokens
	// and handle stop sequences properly
	var decodedText string

	// In actual implementation, this should:
	// 1. Process tokens through stop decoder (stop_decoder.process_tokens())
	// 2. Accumulate text from SequenceDecoderOutput::Text
	// 3. Stop on SequenceDecoderOutput::Stopped or StoppedWithText
	// 4. Flush remaining text from stop decoder

	// Placeholder: Simple tokenizer decode (if available)
	// This is a simplified version - full implementation requires stop decoder
	if components != nil && components.Tokenizer != nil {
		if tok, ok := components.Tokenizer.(tokenizer.Tokenizer); ok {
			// Direct decode (without stop sequence handling)
			// TODO: Integrate with stop decoder for proper stop sequence handling
			text, err := tok.Decode(tokenIDs, true) // skipSpecialTokens=true
			if err != nil {
				rp.logger.Warn("Tokenizer decode failed, using placeholder",
					zap.Error(err),
					zap.Int("token_count", len(tokenIDs)),
				)
			} else {
				decodedText = text
				rp.logger.Debug("Tokens decoded successfully",
					zap.Int("token_count", len(tokenIDs)),
					zap.Int("text_length", len(decodedText)),
				)
			}
		}
	}

	// Fallback: If tokenizer decode failed or not available, try stop decoder
	if decodedText == "" {
		// If we have a stop decoder, use it
		if stopDecoder != nil {
			if decoder, ok := stopDecoder.(tokenizer.StopSequenceDecoder); ok {
				// Process tokens through stop decoder
				decoder.Reset() // Reset for this response
				outputs, err := decoder.ProcessTokens(tokenIDs)
				if err == nil {
					// Accumulate text from outputs
					for _, output := range outputs {
						switch output.Type {
						case tokenizer.OutputTypeText:
							decodedText += output.Text
						case tokenizer.OutputTypeStoppedWithText:
							decodedText += output.Text
							// Stop sequence matched - break
							break
						case tokenizer.OutputTypeStopped:
							// Stop sequence matched - break
							break
						case tokenizer.OutputTypeHeld:
							// Continue processing
						}
					}
					// Flush remaining text
					flushOutput := decoder.Flush()
					if flushOutput.Type == tokenizer.OutputTypeText {
						decodedText += flushOutput.Text
					}
				}
			}
		}

		// If still no decoded text, use placeholder
		if decodedText == "" {
			rp.logger.Warn("Token decoding using placeholder - requires tokenizer and stop decoder",
				zap.Int("token_count", len(tokenIDs)),
			)
			decodedText = fmt.Sprintf("[Decoded text (placeholder - %d tokens)]", len(tokenIDs))
		}
	}

	// TODO: Apply stop sequence trimming if needed
	// This should be handled by stop decoder, but we might need additional trimming

	return decodedText, nil
}
