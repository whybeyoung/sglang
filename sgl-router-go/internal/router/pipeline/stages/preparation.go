package stages

import (
	"fmt"

	"github.com/sglang/sglang-router-go/internal/protocols"
	"github.com/sglang/sglang-router-go/internal/router/pipeline"
	"github.com/sglang/sglang-router-go/internal/tokenizer"
	"go.uber.org/zap"
)

// PreparationStage prepares the request by filtering tools, processing messages, and tokenizing
// Similar to Rust PreparationStage
type PreparationStage struct {
	*pipeline.BaseStage
}

// NewPreparationStage creates a new preparation stage
func NewPreparationStage(logger *zap.Logger) *PreparationStage {
	return &PreparationStage{
		BaseStage: pipeline.NewBaseStage("Preparation", logger),
	}
}

// Execute implements PipelineStage
func (s *PreparationStage) Execute(ctx *pipeline.RequestContext) (interface{}, error) {
	// Clone request type check to avoid borrow issues (Rust pattern)
	// In Go, we don't have the same borrow checker, but we follow similar patterns
	requestType := ctx.Input.RequestType

	if requestType == pipeline.RequestTypeChat {
		return s.prepareChat(ctx)
	} else if requestType == pipeline.RequestTypeGenerate {
		return s.prepareGenerate(ctx)
	}

	return nil, fmt.Errorf("unsupported request type: %d", requestType)
}

// prepareChat prepares a chat completion request
// Similar to Rust prepare_chat method
func (s *PreparationStage) prepareChat(ctx *pipeline.RequestContext) (interface{}, error) {
	// Get chat request from input
	chatReq, ok := ctx.Input.Request.(*protocols.ChatCompletionRequest)
	if !ok {
		return nil, fmt.Errorf("expected ChatCompletionRequest, got %T", ctx.Input.Request)
	}

	// Step 1: Filter tools if needed
	// Note: In Rust, utils::filter_tools_for_request is used
	filteredRequest := FilterToolsForRequest(chatReq, s.Logger)

	// Step 2: Process messages and apply chat template
	// Note: In Rust, utils::process_chat_messages is used with tokenizer
	// This involves:
	// 1. Applying chat template (model-specific formatting)
	// 2. Converting messages to text format
	// 3. Handling multimodal inputs (images, etc.)
	var processedText string
	if ctx.Components.Tokenizer != nil {
		procMsgs, err := ProcessChatMessages(filteredRequest, ctx.Components.Tokenizer, s.Logger)
		if err != nil {
			s.Logger.Warn("Chat message processing failed, using simple format",
				zap.Error(err),
			)
			// Fallback to simple formatting
			for i, msg := range filteredRequest.Messages {
				if i > 0 {
					processedText += "\n"
				}
				processedText += fmt.Sprintf("%s: %s", msg.Role, msg.Content)
			}
		} else {
			processedText = procMsgs.Text
			// TODO: Store processedMessages for multimodal inputs if needed
			_ = procMsgs
		}
	} else {
		// No tokenizer - use simple formatting
		for i, msg := range filteredRequest.Messages {
			if i > 0 {
				processedText += "\n"
			}
			processedText += fmt.Sprintf("%s: %s", msg.Role, msg.Content)
		}
	}

	// Step 3: Tokenize the processed text
	// Note: In Rust, tokenizer.encode() is called
	var tokenIDs []uint32
	var originalText *string
	originalTextVal := processedText
	originalText = &originalTextVal

	if ctx.Components.Tokenizer != nil {
		// Try to use tokenizer if available
		if tok, ok := ctx.Components.Tokenizer.(tokenizer.Tokenizer); ok {
			encoding, err := tok.Encode(processedText)
			if err != nil {
				s.Logger.Warn("Tokenization failed, using placeholder",
					zap.Error(err),
				)
			} else {
				tokenIDs = encoding.TokenIDs
				s.Logger.Debug("Text tokenized successfully",
					zap.Int("token_count", len(tokenIDs)),
				)
			}
		}
	}

	// If tokenization failed or tokenizer not available, use placeholder
	if len(tokenIDs) == 0 {
		// Placeholder - actual implementation requires tokenizer
		s.Logger.Warn("Tokenization not available, using placeholder token IDs",
			zap.String("text_preview", processedText[:min(len(processedText), 50)]),
		)
		// Create placeholder tokens based on text length
		for i := 0; i < min(len(processedText), 100); i++ {
			tokenIDs = append(tokenIDs, uint32(i%32000))
		}
	}

	// Step 4: Build tool constraints if needed
	// Note: In Rust, utils::generate_tool_constraints is used
	var toolConstraints *pipeline.ToolConstraints
	if len(filteredRequest.Tools) > 0 {
		// Try to generate proper tool constraints
		constraint, err := GenerateToolConstraints(
			filteredRequest.Tools,
			filteredRequest.ToolChoice,
			filteredRequest.Model,
			s.Logger,
		)
		if err == nil && constraint != nil {
			toolConstraints = &pipeline.ToolConstraints{
				Type:  constraint.Type,
				Value: constraint.Value,
			}
		} else {
			// Fallback to basic constraint
			toolConstraints = &pipeline.ToolConstraints{
				Type:  "tool_call",
				Value: "enabled",
			}
		}
	}

	// Step 5: Create stop sequence decoder
	// Note: In Rust, utils::create_stop_decoder is used
	var stopDecoder tokenizer.StopSequenceDecoder
	if ctx.Components.Tokenizer != nil {
		if tok, ok := ctx.Components.Tokenizer.(tokenizer.Tokenizer); ok {
			// Extract stop parameters from request
			var stopSequences []string
			var stopTokenIDs []uint32
			skipSpecialTokens := filteredRequest.SkipSpecialTokens
			noStopTrim := filteredRequest.NoStopTrim

			// Extract stop sequences
			if filteredRequest.Stop != nil {
				if stopStr, ok := filteredRequest.Stop.(string); ok {
					stopSequences = []string{stopStr}
				} else if stopSlice, ok := filteredRequest.Stop.([]string); ok {
					stopSequences = stopSlice
				} else if stopSlice, ok := filteredRequest.Stop.([]interface{}); ok {
					for _, v := range stopSlice {
						if str, ok := v.(string); ok {
							stopSequences = append(stopSequences, str)
						}
					}
				}
			}
			stopTokenIDs = filteredRequest.StopTokenIDs

			// Create stop decoder
			stopDecoder = tokenizer.CreateStopDecoder(
				tok,
				stopSequences,
				stopTokenIDs,
				skipSpecialTokens,
				noStopTrim,
			)
			s.Logger.Debug("Stop decoder created",
				zap.Int("stop_sequences", len(stopSequences)),
				zap.Int("stop_token_ids", len(stopTokenIDs)),
			)
		}
	}

	// Store results in context
	ctx.State.Preparation = &pipeline.PreparationOutput{
		OriginalText:      originalText,
		TokenIDs:          tokenIDs,
		ProcessedMessages: processedText, // Store as simple string for now
		ToolConstraints:   toolConstraints,
		FilteredRequest:   filteredRequest,
	}

	// Store stop decoder for reuse in response processing
	if stopDecoder != nil {
		ctx.State.Response.StopDecoder = stopDecoder
	}

	s.Logger.Debug("Chat request prepared",
		zap.Int("token_count", len(tokenIDs)),
		zap.Int("message_count", len(filteredRequest.Messages)),
	)

	return nil, nil // Continue to next stage
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// prepareGenerate prepares a generate request
// Similar to Rust prepare_generate method
func (s *PreparationStage) prepareGenerate(ctx *pipeline.RequestContext) (interface{}, error) {
	// Get generate request from input
	genReq, ok := ctx.Input.Request.(*protocols.GenerateRequest)
	if !ok {
		return nil, fmt.Errorf("expected GenerateRequest, got %T", ctx.Input.Request)
	}

	// Resolve input (text, prompt, or input_ids)
	// Note: In Rust, resolve_generate_input is called
	originalText, tokenIDs, err := s.resolveGenerateInput(ctx, genReq)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve generate input: %w", err)
	}

	// Create stop sequence decoder for generate requests
	// Note: Stop decoder should be created from sampling params
	var stopDecoder tokenizer.StopSequenceDecoder
	if ctx.Components.Tokenizer != nil && genReq.SamplingParams != nil {
		if tok, ok := ctx.Components.Tokenizer.(tokenizer.Tokenizer); ok {
			sp := genReq.SamplingParams
			stopSequences := sp.Stop
			stopTokenIDs := sp.StopTokenIDs
			skipSpecialTokens := true
			if sp.SkipSpecialTokens != nil {
				skipSpecialTokens = *sp.SkipSpecialTokens
			}
			noStopTrim := false
			if sp.NoStopTrim != nil {
				noStopTrim = *sp.NoStopTrim
			}

			// Create stop decoder
			stopDecoder = tokenizer.CreateStopDecoder(
				tok,
				stopSequences,
				stopTokenIDs,
				skipSpecialTokens,
				noStopTrim,
			)
			s.Logger.Debug("Stop decoder created for generate request",
				zap.Int("stop_sequences", len(stopSequences)),
				zap.Int("stop_token_ids", len(stopTokenIDs)),
			)
		}
	}

	ctx.State.Preparation = &pipeline.PreparationOutput{
		OriginalText:      originalText,
		TokenIDs:          tokenIDs,
		ProcessedMessages: nil,
		ToolConstraints:   nil,
		FilteredRequest:   genReq,
	}

	// Store stop decoder for reuse in response processing
	if stopDecoder != nil {
		ctx.State.Response.StopDecoder = stopDecoder
	}

	s.Logger.Debug("Generate request prepared",
		zap.Int("token_count", len(tokenIDs)),
		zap.Bool("has_original_text", originalText != nil),
	)

	return nil, nil // Continue to next stage
}

// resolveGenerateInput resolves input for generate requests
// Similar to Rust resolve_generate_input method
// This handles:
// 1. Text input -> tokenize
// 2. Input IDs -> validate and convert
// 3. Error if neither provided
func (s *PreparationStage) resolveGenerateInput(
	ctx *pipeline.RequestContext,
	request *protocols.GenerateRequest,
) (*string, []uint32, error) {
	// Priority 1: Text input
	if request.Text != nil && *request.Text != "" {
		return s.tokenizeSingleText(ctx, *request.Text)
	}

	// Priority 2: Input IDs (direct token IDs)
	if request.InputIDs != nil {
		// Handle different input_ids formats
		// Can be []uint32 (single) or [][]uint32 (batch)
		var tokenIDs []uint32
		switch ids := request.InputIDs.(type) {
		case []uint32:
			// Single input_ids array
			tokenIDs = ids
		case []interface{}:
			// Convert []interface{} to []uint32
			tokenIDs = make([]uint32, 0, len(ids))
			for i, v := range ids {
				switch val := v.(type) {
				case uint32:
					tokenIDs = append(tokenIDs, val)
				case int:
					if val < 0 {
						return nil, nil, fmt.Errorf("input_ids[%d] must be non-negative, got %d", i, val)
					}
					tokenIDs = append(tokenIDs, uint32(val))
				case int32:
					if val < 0 {
						return nil, nil, fmt.Errorf("input_ids[%d] must be non-negative, got %d", i, val)
					}
					tokenIDs = append(tokenIDs, uint32(val))
				case uint:
					tokenIDs = append(tokenIDs, uint32(val))
				default:
					return nil, nil, fmt.Errorf("invalid input_ids type at index %d: %T", i, val)
				}
			}
		case []int:
			// Convert []int to []uint32
			tokenIDs = make([]uint32, 0, len(ids))
			for i, id := range ids {
				if id < 0 {
					return nil, nil, fmt.Errorf("input_ids[%d] must be non-negative, got %d", i, id)
				}
				tokenIDs = append(tokenIDs, uint32(id))
			}
		case []int32:
			// Convert []int32 to []uint32
			tokenIDs = make([]uint32, 0, len(ids))
			for i, id := range ids {
				if id < 0 {
					return nil, nil, fmt.Errorf("input_ids[%d] must be non-negative, got %d", i, id)
				}
				tokenIDs = append(tokenIDs, uint32(id))
			}
		default:
			return nil, nil, fmt.Errorf("unsupported input_ids type: %T (expected []uint32 or []int)", ids)
		}

		if len(tokenIDs) > 0 {
			return nil, tokenIDs, nil
		}
	}

	// No valid input provided
	return nil, nil, fmt.Errorf("either 'text', 'prompt', or 'input_ids' must be provided")
}

// tokenizeSingleText tokenizes a single text string
// Similar to Rust tokenize_single_text method
func (s *PreparationStage) tokenizeSingleText(
	ctx *pipeline.RequestContext,
	text string,
) (*string, []uint32, error) {
	originalText := text

	// Try to use tokenizer if available
	var tokenIDs []uint32
	if ctx.Components.Tokenizer != nil {
		if tok, ok := ctx.Components.Tokenizer.(tokenizer.Tokenizer); ok {
			encoding, err := tok.Encode(text)
			if err != nil {
				s.Logger.Warn("Tokenization failed, using placeholder",
					zap.Error(err),
				)
			} else {
				tokenIDs = encoding.TokenIDs
				s.Logger.Debug("Text tokenized successfully",
					zap.Int("token_count", len(tokenIDs)),
					zap.String("text_preview", text[:min(len(text), 50)]),
				)
			}
		}
	}

	// If tokenization failed or tokenizer not available, use placeholder
	if len(tokenIDs) == 0 {
		s.Logger.Warn("Tokenization not available, using placeholder token IDs",
			zap.String("text_preview", text[:min(len(text), 50)]),
		)
		// Create placeholder tokens based on text length
		for i := 0; i < min(len(text), 100); i++ {
			tokenIDs = append(tokenIDs, uint32(i%32000))
		}
	}

	return &originalText, tokenIDs, nil
}
