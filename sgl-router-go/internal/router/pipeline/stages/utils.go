package stages

import (
	"encoding/json"
	"fmt"

	"github.com/sglang/sglang-router-go/internal/protocols"
	"github.com/sglang/sglang-router-go/internal/tokenizer"
	"go.uber.org/zap"
)

// FilterToolsForRequest filters tools based on tool_choice
// Similar to Rust utils::filter_tools_for_request
// Returns a filtered copy of the request if filtering is needed
func FilterToolsForRequest(
	request *protocols.ChatCompletionRequest,
	logger *zap.Logger,
) *protocols.ChatCompletionRequest {
	if request.ToolChoice == nil || len(request.Tools) == 0 {
		return request
	}

	// Parse tool_choice JSON to determine type
	toolChoiceJSON, err := json.Marshal(request.ToolChoice)
	if err != nil {
		logger.Warn("Failed to marshal tool_choice, skipping filtering",
			zap.Error(err),
		)
		return request
	}

	var toolChoiceMap map[string]interface{}
	if err := json.Unmarshal(toolChoiceJSON, &toolChoiceMap); err != nil {
		// If it's a string (like "none", "auto", "required"), no filtering needed
		if str, ok := request.ToolChoice.(string); ok {
			if str == "none" || str == "auto" || str == "required" {
				return request
			}
		}
		logger.Warn("Failed to parse tool_choice, skipping filtering",
			zap.Error(err),
		)
		return request
	}

	// Check for AllowedTools
	if toolType, ok := toolChoiceMap["type"].(string); ok && toolType == "allowed_tools" {
		if toolsValue, ok := toolChoiceMap["tools"]; ok {
			if toolsArray, ok := toolsValue.([]interface{}); ok {
				// Build set of allowed tool names
				allowedNames := make(map[string]bool)
				for _, toolRef := range toolsArray {
					if toolMap, ok := toolRef.(map[string]interface{}); ok {
						if name, ok := toolMap["name"].(string); ok {
							allowedNames[name] = true
						}
					}
				}

				// Filter tools
				if len(allowedNames) > 0 {
					filteredTools := make([]protocols.Tool, 0)
					for _, tool := range request.Tools {
						if allowedNames[tool.Function.Name] {
							filteredTools = append(filteredTools, tool)
						}
					}

					// Return filtered copy
					filteredRequest := *request
					filteredRequest.Tools = filteredTools
					logger.Debug("Filtered tools based on allowed_tools",
						zap.Int("original_count", len(request.Tools)),
						zap.Int("filtered_count", len(filteredTools)),
					)
					return &filteredRequest
				}
			}
		}
	}

	// Check for Function
	if toolType, ok := toolChoiceMap["type"].(string); ok && toolType == "function" {
		if functionValue, ok := toolChoiceMap["function"]; ok {
			if functionMap, ok := functionValue.(map[string]interface{}); ok {
				if functionName, ok := functionMap["name"].(string); ok {
					// Filter to specific function
					filteredTools := make([]protocols.Tool, 0)
					for _, tool := range request.Tools {
						if tool.Function.Name == functionName {
							filteredTools = append(filteredTools, tool)
						}
					}

					if len(filteredTools) > 0 {
						filteredRequest := *request
						filteredRequest.Tools = filteredTools
						logger.Debug("Filtered tools to specific function",
							zap.String("function_name", functionName),
							zap.Int("filtered_count", len(filteredTools)),
						)
						return &filteredRequest
					}
				}
			}
		}
	}

	// No filtering needed
	return request
}

// GenerateToolConstraints generates tool constraints from tool choice
// Similar to Rust utils::generate_tool_constraints
// Returns (constraint_type, constraint_value) or nil if no constraint
func GenerateToolConstraints(
	tools []protocols.Tool,
	toolChoice interface{},
	model string,
	logger *zap.Logger,
) (*ToolConstraintResult, error) {
	if len(tools) == 0 {
		return nil, nil
	}

	if toolChoice == nil {
		return nil, nil
	}

	// Parse tool_choice JSON to determine type
	toolChoiceJSON, err := json.Marshal(toolChoice)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal tool_choice: %w", err)
	}

	// Try to parse as string first (for "none", "auto", "required")
	var toolChoiceStr string
	if err := json.Unmarshal(toolChoiceJSON, &toolChoiceStr); err == nil {
		// It's a string value
		if toolChoiceStr == "required" {
			// Required: Array of tool calls with minItems: 1
			schema, err := buildRequiredArraySchema(tools, logger)
			if err != nil {
				return nil, err
			}
			return &ToolConstraintResult{
				Type:  "json_schema",
				Value: schema,
			}, nil
		}
		// "auto" or "none" - no constraint
		return nil, nil
	}

	// Try to parse as object
	var toolChoiceMap map[string]interface{}
	if err := json.Unmarshal(toolChoiceJSON, &toolChoiceMap); err != nil {
		return nil, fmt.Errorf("failed to parse tool_choice: %w", err)
	}

	toolType, _ := toolChoiceMap["type"].(string)

	// Handle Function choice
	if toolType == "function" {
		if functionValue, ok := toolChoiceMap["function"].(map[string]interface{}); ok {
			if len(tools) == 0 {
				return nil, nil
			}
			// Find the matching tool
			var matchingTool *protocols.Tool
			if functionName, ok := functionValue["name"].(string); ok {
				for i := range tools {
					if tools[i].Function.Name == functionName {
						matchingTool = &tools[i]
						break
					}
				}
			}

			if matchingTool != nil {
				// Return the tool's parameters schema directly (not wrapped in array)
				paramsSchema := string(matchingTool.Function.Parameters)
				return &ToolConstraintResult{
					Type:  "json_schema",
					Value: paramsSchema,
				}, nil
			}
		}
		return nil, nil
	}

	// Handle AllowedTools choice
	if toolType == "allowed_tools" {
		if mode, ok := toolChoiceMap["mode"].(string); ok && mode == "required" {
			if len(tools) == 0 {
				return nil, nil
			}
			// Build required array schema
			schema, err := buildRequiredArraySchema(tools, logger)
			if err != nil {
				return nil, err
			}
			return &ToolConstraintResult{
				Type:  "json_schema",
				Value: schema,
			}, nil
		}
		// "auto" mode - no constraint
		return nil, nil
	}

	// Unknown type or no constraint needed
	return nil, nil
}

// buildRequiredArraySchema builds JSON schema for required tool calls (array with minItems: 1)
// Similar to Rust build_required_array_schema
func buildRequiredArraySchema(tools []protocols.Tool, logger *zap.Logger) (string, error) {
	// Build anyOf schemas for each tool
	anyOfSchemas := make([]map[string]interface{}, 0, len(tools))
	for _, tool := range tools {
		// Parse tool parameters
		var paramsSchema map[string]interface{}
		if err := json.Unmarshal(tool.Function.Parameters, &paramsSchema); err != nil {
			logger.Warn("Failed to parse tool parameters schema",
				zap.String("tool_name", tool.Function.Name),
				zap.Error(err),
			)
			continue
		}

		toolSchema := map[string]interface{}{
			"properties": map[string]interface{}{
				"name": map[string]interface{}{
					"type": "string",
					"enum": []string{tool.Function.Name},
				},
				"parameters": paramsSchema,
			},
			"required": []string{"name", "parameters"},
		}
		anyOfSchemas = append(anyOfSchemas, toolSchema)
	}

	if len(anyOfSchemas) == 0 {
		return "", fmt.Errorf("no valid tool schemas found")
	}

	// Consolidate $defs from all tools
	allDefs := make(map[string]interface{})
	for _, tool := range tools {
		var paramsSchema map[string]interface{}
		if err := json.Unmarshal(tool.Function.Parameters, &paramsSchema); err != nil {
			continue
		}

		if defsValue, ok := paramsSchema["$defs"]; ok {
			if defsMap, ok := defsValue.(map[string]interface{}); ok {
				for defName, defSchema := range defsMap {
					if existing, exists := allDefs[defName]; exists {
						// Check for conflicts
						existingJSON, _ := json.Marshal(existing)
						newJSON, _ := json.Marshal(defSchema)
						if string(existingJSON) != string(newJSON) {
							return "", fmt.Errorf(
								"tool definition '%s' has multiple conflicting schemas, which is not supported",
								defName,
							)
						}
					} else {
						allDefs[defName] = defSchema
					}
				}
			}
		}
	}

	// Build the full array schema
	arraySchema := map[string]interface{}{
		"type":     "array",
		"minItems": 1,
		"items": map[string]interface{}{
			"type":  "object",
			"anyOf": anyOfSchemas,
		},
	}

	// Add $defs if any were found
	if len(allDefs) > 0 {
		arraySchema["$defs"] = allDefs
	}

	// Serialize to JSON string
	schemaJSON, err := json.Marshal(arraySchema)
	if err != nil {
		return "", fmt.Errorf("failed to serialize tool schema: %w", err)
	}

	return string(schemaJSON), nil
}

// ToolConstraintResult represents the result of tool constraint generation
type ToolConstraintResult struct {
	Type  string // e.g., "json_schema"
	Value string // JSON schema string
}

// ProcessChatMessages processes chat messages and applies chat template
// Similar to Rust utils::process_chat_messages
func ProcessChatMessages(
	request *protocols.ChatCompletionRequest,
	tok interface{}, // Tokenizer interface
	logger *zap.Logger,
) (*ProcessedMessages, error) {
	// Try to use HuggingFace tokenizer with chat template
	var chatTemplate *string
	var contentFormat tokenizer.ChatTemplateContentFormat

	hfTokenizer, ok := tok.(*tokenizer.HuggingFaceTokenizer)
	if ok && hfTokenizer != nil {
		// Use HuggingFace tokenizer's chat template
		chatTemplate = hfTokenizer.ChatTemplate()
		contentFormat = hfTokenizer.ChatTemplateContentFormat()
	} else {
		// Try Rust FFI tokenizer
		rustTokenizer, ok := tok.(*tokenizer.RustFFITokenizer)
		if ok && rustTokenizer != nil {
			chatTemplate = rustTokenizer.ChatTemplate()
			contentFormat = rustTokenizer.ChatTemplateContentFormat()
		}
	}

	// If no chat template available, use simple formatting
	if chatTemplate == nil {
		logger.Debug("No chat template available, using simple message formatting",
			zap.String("tokenizer_type", fmt.Sprintf("%T", tok)),
		)
		return processChatMessagesSimple(request)
	}

	// contentFormat already set above (from hfTokenizer or rustTokenizer)

	// Convert messages to map format for template processing
	messagesJSON, err := convertMessagesToJSON(request.Messages)
	if err != nil {
		return nil, fmt.Errorf("failed to convert messages to JSON: %w", err)
	}

	// Process content format transformation
	transformedMessages, err := tokenizer.ProcessContentFormat(messagesJSON, contentFormat)
	if err != nil {
		return nil, fmt.Errorf("failed to process content format: %w", err)
	}

	// Process tool call arguments (if any)
	// TODO: Implement process_tool_call_arguments equivalent
	// For now, skip tool call processing

	// Convert to []map[string]interface{} for template processor
	messagesForTemplate := make([]map[string]interface{}, len(transformedMessages))
	for i, msg := range transformedMessages {
		messagesForTemplate[i] = msg
	}

	// Apply chat template
	templateProcessor := tokenizer.NewChatTemplateProcessor(*chatTemplate)
	params := tokenizer.ChatTemplateParams{
		AddGenerationPrompt: true, // Default to true
		Tools:               nil,  // TODO: Convert tools to JSON if needed
	}

	formattedText, err := templateProcessor.ApplyChatTemplate(messagesForTemplate, params)
	if err != nil {
		logger.Warn("Failed to apply chat template, using simple formatting",
			zap.Error(err),
		)
		return processChatMessagesSimple(request)
	}

	return &ProcessedMessages{
		Text:             formattedText,
		MultimodalInputs: nil, // TODO: Extract multimodal inputs if needed
	}, nil
}

// processChatMessagesSimple provides simple message formatting as fallback
func processChatMessagesSimple(request *protocols.ChatCompletionRequest) (*ProcessedMessages, error) {
	var text string
	for i, msg := range request.Messages {
		if i > 0 {
			text += "\n"
		}
		// Simple formatting
		contentStr := extractContentString(msg.Content)
		text += fmt.Sprintf("%s: %s", msg.Role, contentStr)
	}

	return &ProcessedMessages{
		Text:             text,
		MultimodalInputs: nil,
	}, nil
}

// extractContentString extracts content as string
func extractContentString(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		// Extract text parts from OpenAI format
		var parts []string
		for _, part := range v {
			if partMap, ok := part.(map[string]interface{}); ok {
				if typ, _ := partMap["type"].(string); typ == "text" {
					if text, _ := partMap["text"].(string); text != "" {
						parts = append(parts, text)
					}
				}
			}
		}
		if len(parts) > 0 {
			return fmt.Sprintf("%v", parts)
		}
		return fmt.Sprintf("%v", v)
	default:
		return fmt.Sprintf("%v", v)
	}
}

// convertMessagesToJSON converts ChatMessage slice to []map[string]interface{}
func convertMessagesToJSON(messages []protocols.ChatMessage) ([]map[string]interface{}, error) {
	result := make([]map[string]interface{}, len(messages))
	for i, msg := range messages {
		// Convert to JSON and back to map
		msgJSON, err := json.Marshal(msg)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal message %d: %w", i, err)
		}

		var msgMap map[string]interface{}
		if err := json.Unmarshal(msgJSON, &msgMap); err != nil {
			return nil, fmt.Errorf("failed to unmarshal message %d: %w", i, err)
		}

		result[i] = msgMap
	}
	return result, nil
}

// ProcessedMessages represents processed chat messages
type ProcessedMessages struct {
	Text             string
	MultimodalInputs interface{} // TODO: Define proper type for multimodal inputs
}
