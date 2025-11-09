// Package tokenizer provides a high-level tokenizer interface
package tokenizer

import (
	"encoding/json"
	"fmt"

	"github.com/sglang/sgl-router-go-client/internal/ffi"
)

// Message represents a chat message
type Message struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

// Tool represents a function tool definition
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function represents a function definition
type Function struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	Parameters  interface{} `json:"parameters"` // JSON Schema
	Strict      *bool       `json:"strict,omitempty"`
}

// Tokenizer wraps the FFI tokenizer with high-level methods
type Tokenizer struct {
	handle *ffi.TokenizerHandle
}

// NewTokenizer creates a new tokenizer from a file path
func NewTokenizer(path string) (*Tokenizer, error) {
	handle, err := ffi.NewTokenizerFromFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to create tokenizer: %w", err)
	}

	return &Tokenizer{handle: handle}, nil
}

// Close closes the tokenizer
func (t *Tokenizer) Close() error {
	if t.handle != nil {
		return t.handle.Close()
	}
	return nil
}

// EncodeText encodes plain text to token IDs
func (t *Tokenizer) EncodeText(text string) ([]uint32, error) {
	return t.handle.Encode(text)
}

// ProcessMessages processes chat messages and returns tokenized input
func (t *Tokenizer) ProcessMessages(messages []Message) ([]uint32, string, error) {
	return t.ProcessMessagesWithTools(messages, nil)
}

// ProcessMessagesWithTools processes chat messages with tools and returns tokenized input
func (t *Tokenizer) ProcessMessagesWithTools(messages []Message, tools []Tool) ([]uint32, string, error) {
	// Convert messages to JSON
	messagesJSON, err := json.Marshal(messages)
	if err != nil {
		return nil, "", fmt.Errorf("failed to marshal messages: %w", err)
	}

	// Convert tools to JSON if provided
	var toolsJSON string
	if tools != nil && len(tools) > 0 {
		toolsJSONBytes, err := json.Marshal(tools)
		if err != nil {
			return nil, "", fmt.Errorf("failed to marshal tools: %w", err)
		}
		toolsJSON = string(toolsJSONBytes)
	}

	// Apply chat template with tools
	var processedText string
	if toolsJSON != "" {
		processedText, err = t.handle.ApplyChatTemplateWithTools(string(messagesJSON), toolsJSON)
	} else {
		processedText, err = t.handle.ApplyChatTemplate(string(messagesJSON))
	}
	if err != nil {
		return nil, "", fmt.Errorf("failed to apply chat template: %w", err)
	}

	// Encode to token IDs
	tokenIds, err := t.handle.Encode(processedText)
	if err != nil {
		return nil, "", fmt.Errorf("failed to encode: %w", err)
	}

	return tokenIds, processedText, nil
}

// ProcessSingleMessage processes a single message (for simple use cases)
func (t *Tokenizer) ProcessSingleMessage(role, content string) ([]uint32, string, error) {
	messages := []Message{
		{Role: role, Content: content},
	}
	return t.ProcessMessages(messages)
}

// DecodeText decodes token IDs back to text
func (t *Tokenizer) DecodeText(tokenIds []uint32, skipSpecialTokens bool) (string, error) {
	return t.handle.Decode(tokenIds, skipSpecialTokens)
}

// GetHandle returns the underlying FFI tokenizer handle
func (t *Tokenizer) GetHandle() *ffi.TokenizerHandle {
	return t.handle
}
