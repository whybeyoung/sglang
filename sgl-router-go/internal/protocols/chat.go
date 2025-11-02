package protocols

import (
	"encoding/json"
)

// ChatMessage represents a single message in a chat conversation
// Similar to Rust ChatMessage enum
type ChatMessage struct {
	Role    string          `json:"role"`    // "system", "user", "assistant", "tool", "function"
	Content json.RawMessage `json:"content"` // Can be string or array of ContentPart
	Name    *string         `json:"name,omitempty"`

	// Assistant-specific fields
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ReasoningContent *string    `json:"reasoning_content,omitempty"`

	// Tool-specific fields
	ToolCallID *string `json:"tool_call_id,omitempty"`
}

// ContentPart represents a part of user message content (text, image, etc.)
type ContentPart struct {
	Type     string          `json:"type"` // "text", "image_url", etc.
	Text     *string         `json:"text,omitempty"`
	ImageURL *ImageURL       `json:"image_url,omitempty"`
	Other    json.RawMessage `json:"-"` // For extensibility
}

// ImageURL represents an image URL
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// ToolCall represents a function/tool call
type ToolCall struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"` // "function"
	Function FunctionCallDefinition `json:"function"`
}

// FunctionCallDefinition represents a function call definition
type FunctionCallDefinition struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"` // JSON string
}

// ChatCompletionRequest represents an OpenAI-compatible chat completion request
// Similar to Rust ChatCompletionRequest
type ChatCompletionRequest struct {
	Messages            []ChatMessage      `json:"messages"`
	Model               string             `json:"model"`
	Temperature         *float32           `json:"temperature,omitempty"`
	TopP                *float32           `json:"top_p,omitempty"`
	TopK                *int32             `json:"top_k,omitempty"`
	N                   *uint32            `json:"n,omitempty"`
	Stream              bool               `json:"stream,omitempty"`
	Stop                interface{}        `json:"stop,omitempty"` // string or []string
	MaxTokens           *uint32            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *uint32            `json:"max_completion_tokens,omitempty"`
	PresencePenalty     *float32           `json:"presence_penalty,omitempty"`
	FrequencyPenalty    *float32           `json:"frequency_penalty,omitempty"`
	LogitBias           map[string]float32 `json:"logit_bias,omitempty"`
	Logprobs            bool               `json:"logprobs,omitempty"`
	User                *string            `json:"user,omitempty"`

	// Tool-related fields
	Tools      []Tool      `json:"tools,omitempty"`
	ToolChoice interface{} `json:"tool_choice,omitempty"` // "none", "auto", "required", or object

	// Response format
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`

	// SGLang extensions
	SkipSpecialTokens bool     `json:"skip_special_tokens,omitempty"`
	NoStopTrim        bool     `json:"no_stop_trim,omitempty"`
	StopTokenIDs      []uint32 `json:"stop_token_ids,omitempty"`
}

// ResponseFormat specifies the format of the response
type ResponseFormat struct {
	Type string `json:"type"` // "text", "json_object"
}

// Tool represents a tool/function definition
type Tool struct {
	Type     string             `json:"type"` // "function"
	Function FunctionDefinition `json:"function"`
}

// FunctionDefinition represents a function definition
type FunctionDefinition struct {
	Name        string          `json:"name"`
	Description *string         `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"` // JSON Schema
}

// ChatCompletionResponse represents an OpenAI-compatible chat completion response
// Similar to Rust ChatCompletionResponse
type ChatCompletionResponse struct {
	ID                string                 `json:"id"`
	Object            string                 `json:"object"` // "chat.completion"
	Created           int64                  `json:"created"`
	Model             string                 `json:"model"`
	Choices           []ChatCompletionChoice `json:"choices"`
	Usage             *Usage                 `json:"usage,omitempty"`
	SystemFingerprint *string                `json:"system_fingerprint,omitempty"`
}

// ChatCompletionChoice represents a single completion choice
type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"` // "stop", "length", "tool_calls", etc.
	Logprobs     *LogProbs   `json:"logprobs,omitempty"`
}

// Usage represents token usage information
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// LogProbs represents log probabilities
type LogProbs struct {
	TokenLogprobs []float32                `json:"token_logprobs"`
	Tokens        []string                 `json:"tokens"`
	TopLogprobs   []map[string]interface{} `json:"top_logprobs,omitempty"`
}

// IsStreaming returns whether the request is for streaming
func (r *ChatCompletionRequest) IsStreaming() bool {
	return r.Stream
}

// GetModel returns the model name
func (r *ChatCompletionRequest) GetModel() string {
	return r.Model
}
