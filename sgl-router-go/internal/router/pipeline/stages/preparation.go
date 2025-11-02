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
	// TODO: Implement tool filtering for Go
	// For now, use original request
	filteredRequest := chatReq

	// Step 2: Process messages and apply chat template
	// Note: In Rust, utils::process_chat_messages is used with tokenizer
	// This involves:
	// 1. Applying chat template (model-specific formatting)
	// 2. Converting messages to text format
	// 3. Handling multimodal inputs (images, etc.)
	// TODO: Implement message processing with tokenizer
	// processedMessages := processChatMessages(filteredRequest, ctx.Components.Tokenizer)

	// For now, create a simple text representation
	var processedText string
	for i, msg := range filteredRequest.Messages {
		if i > 0 {
			processedText += "\n"
		}
		// Simple formatting - actual implementation should use chat template
		processedText += fmt.Sprintf("%s: %s", msg.Role, msg.Content)
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
	// TODO: Implement tool constraint generation
	var toolConstraints *pipeline.ToolConstraints
	if len(filteredRequest.Tools) > 0 {
		// Basic tool constraint (actual implementation should be more sophisticated)
		toolConstraints = &pipeline.ToolConstraints{
			Type:  "tool_call",
			Value: "enabled",
		}
	}

	// Step 5: Create stop sequence decoder
	// Note: In Rust, utils::create_stop_decoder is used
	// TODO: Implement stop decoder
	// stopDecoder := createStopDecoder(ctx.Components.Tokenizer, chatReq.Stop, ...)

	// Store results in context
	ctx.State.Preparation = &pipeline.PreparationOutput{
		OriginalText:      originalText,
		TokenIDs:          tokenIDs,
		ProcessedMessages: processedText, // Store as simple string for now
		ToolConstraints:   toolConstraints,
		FilteredRequest:   filteredRequest,
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
	// TODO: Implement stop decoder creation
	// stopDecoder := createStopDecoder(ctx.Components.Tokenizer, ...)

	ctx.State.Preparation = &pipeline.PreparationOutput{
		OriginalText:      originalText,
		TokenIDs:          tokenIDs,
		ProcessedMessages: nil,
		ToolConstraints:   nil,
		FilteredRequest:   genReq,
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
