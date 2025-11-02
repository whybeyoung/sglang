package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/flosch/pongo2/v6"
)

// ChatTemplateContentFormat represents the expected content format
// Similar to Rust ChatTemplateContentFormat
type ChatTemplateContentFormat int

const (
	// ChatTemplateContentFormatString expects simple string content
	ChatTemplateContentFormatString ChatTemplateContentFormat = iota
	// ChatTemplateContentFormatOpenAI expects structured content (list of parts)
	ChatTemplateContentFormatOpenAI
)

// String returns string representation
func (f ChatTemplateContentFormat) String() string {
	switch f {
	case ChatTemplateContentFormatOpenAI:
		return "openai"
	default:
		return "string"
	}
}

// ChatTemplateParams contains parameters for chat template application
// Similar to Rust ChatTemplateParams
type ChatTemplateParams struct {
	AddGenerationPrompt bool
	Tools               interface{} // []map[string]interface{} or nil
	Documents           interface{} // []map[string]interface{} or nil
	TemplateKwargs      map[string]interface{}
}

// ChatTemplateProcessor processes chat templates using Jinja-like syntax
// Similar to Rust ChatTemplateProcessor
// NOTE: Full Jinja support requires a Jinja library
// For now, we'll implement a simplified version or use text/template
type ChatTemplateProcessor struct {
	template string
}

// NewChatTemplateProcessor creates a new chat template processor
func NewChatTemplateProcessor(template string) *ChatTemplateProcessor {
	return &ChatTemplateProcessor{
		template: template,
	}
}

// ApplyChatTemplate applies the chat template to messages using pongo2 (Jinja2-compatible)
// Similar to Rust ChatTemplateProcessor::apply_chat_template
// Uses pongo2 library for full Jinja2 template rendering support
func (p *ChatTemplateProcessor) ApplyChatTemplate(
	messages []map[string]interface{},
	params ChatTemplateParams,
) (string, error) {
	// Try to parse and render using pongo2 (full Jinja2 support)
	output, err := p.renderWithPongo2(messages, params)
	if err != nil {
		// If pongo2 fails, fall back to simple rendering
		return p.renderTemplateSimple(messages, params)
	}
	return output, nil
}

// renderWithPongo2 renders template using pongo2 library
func (p *ChatTemplateProcessor) renderWithPongo2(
	messages []map[string]interface{},
	params ChatTemplateParams,
) (string, error) {
	// Parse template using pongo2
	tpl, err := pongo2.FromString(p.template)
	if err != nil {
		return "", fmt.Errorf("failed to parse chat template: %w", err)
	}

	// Build context for template rendering
	// Note: pongo2.Context is a map[string]interface{}
	ctx := pongo2.Context{
		"messages":              messages,
		"add_generation_prompt": params.AddGenerationPrompt,
	}

	// Add optional parameters
	if params.Tools != nil {
		ctx["tools"] = params.Tools
	}
	if params.Documents != nil {
		ctx["documents"] = params.Documents
	}
	if params.TemplateKwargs != nil {
		// Merge template_kwargs into context
		for k, v := range params.TemplateKwargs {
			ctx[k] = v
		}
	}

	// Render template
	output, err := tpl.Execute(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to render chat template: %w", err)
	}

	return output, nil
}

// renderTemplateSimple provides fallback simple template rendering
// This is used as a fallback if pongo2 parsing fails
func (p *ChatTemplateProcessor) renderTemplateSimple(
	messages []map[string]interface{},
	params ChatTemplateParams,
) (string, error) {
	var result strings.Builder

	// Simple formatting fallback
	for _, msg := range messages {
		role, _ := msg["role"].(string)
		content := extractContent(msg["content"])
		result.WriteString(formatMessageSimple(role, content))
	}

	return result.String(), nil
}

// extractContent extracts content from message
func extractContent(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		// OpenAI format - extract text parts
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
		return strings.Join(parts, " ")
	default:
		return fmt.Sprintf("%v", v)
	}
}

// formatMessageSimple formats a message simply
// This is a placeholder - actual implementation needs Jinja template rendering
func formatMessageSimple(role, content string) string {
	// Very basic formatting - actual implementation would use template
	switch role {
	case "system":
		return fmt.Sprintf("System: %s\n", content)
	case "user":
		return fmt.Sprintf("User: %s\n", content)
	case "assistant":
		return fmt.Sprintf("Assistant: %s\n", content)
	default:
		return fmt.Sprintf("%s: %s\n", role, content)
	}
}

// DetectChatTemplateContentFormat detects the content format expected by a template
// Similar to Rust detect_chat_template_content_format
// This is a simplified version - full implementation would parse AST
func DetectChatTemplateContentFormat(template string) ChatTemplateContentFormat {
	// Simple heuristics:
	// - If template contains iteration over content parts, it's OpenAI format
	// - Otherwise, it's String format

	// Check for OpenAI-style patterns
	// Simple substring checks for common patterns
	if strings.Contains(template, "content") && strings.Contains(template, "for") {
		// Check for iteration over content
		if strings.Contains(template, "for") && strings.Contains(template, "in") {
			if strings.Contains(template, "content") || strings.Contains(template, "message.content") {
				return ChatTemplateContentFormatOpenAI
			}
		}
	}
	if strings.Contains(template, "content is sequence") || strings.Contains(template, "content|length") {
		return ChatTemplateContentFormatOpenAI
	}

	// Default to String format
	return ChatTemplateContentFormatString
}

// LoadChatTemplateFromConfig loads chat template from tokenizer_config.json
// Similar to Rust load_chat_template_from_config
func LoadChatTemplateFromConfig(configPath string) (*string, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config map[string]interface{}
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config JSON: %w", err)
	}

	// Look for chat_template field
	if templateValue, ok := config["chat_template"]; ok {
		if templateStr, ok := templateValue.(string); ok {
			return &templateStr, nil
		}
	}

	return nil, nil
}

// ProcessContentFormat processes messages based on content format
// Similar to Rust process_content_format
func ProcessContentFormat(
	messages []map[string]interface{},
	contentFormat ChatTemplateContentFormat,
) ([]map[string]interface{}, error) {
	result := make([]map[string]interface{}, len(messages))

	for i, msg := range messages {
		// Clone message
		processedMsg := make(map[string]interface{})
		for k, v := range msg {
			processedMsg[k] = v
		}

		// Transform content field if present
		if content, ok := processedMsg["content"]; ok {
			transformed := transformContentField(content, contentFormat)
			processedMsg["content"] = transformed
		}

		result[i] = processedMsg
	}

	return result, nil
}

// transformContentField transforms a content field based on format
// Similar to Rust transform_content_field
func transformContentField(content interface{}, format ChatTemplateContentFormat) interface{} {
	// Check if content is an array (multimodal)
	contentArray, ok := content.([]interface{})
	if !ok {
		// Not multimodal, return as-is
		return content
	}

	switch format {
	case ChatTemplateContentFormatString:
		// Extract and join text parts only
		var textParts []string
		for _, part := range contentArray {
			if partMap, ok := part.(map[string]interface{}); ok {
				if typ, _ := partMap["type"].(string); typ == "text" {
					if text, _ := partMap["text"].(string); text != "" {
						textParts = append(textParts, text)
					}
				}
			}
		}
		if len(textParts) > 0 {
			return strings.Join(textParts, " ")
		}
		return content

	case ChatTemplateContentFormatOpenAI:
		// Replace media URLs with simple type placeholders
		processedParts := make([]interface{}, 0, len(contentArray))
		for _, part := range contentArray {
			if partMap, ok := part.(map[string]interface{}); ok {
				if typ, _ := partMap["type"].(string); typ != "" {
					switch typ {
					case "image_url":
						processedParts = append(processedParts, map[string]interface{}{
							"type": "image",
						})
					case "video_url":
						processedParts = append(processedParts, map[string]interface{}{
							"type": "video",
						})
					case "audio_url":
						processedParts = append(processedParts, map[string]interface{}{
							"type": "audio",
						})
					default:
						processedParts = append(processedParts, part)
					}
				} else {
					processedParts = append(processedParts, part)
				}
			} else {
				processedParts = append(processedParts, part)
			}
		}
		return processedParts

	default:
		return content
	}
}
