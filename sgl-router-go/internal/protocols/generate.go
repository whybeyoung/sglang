package protocols

import (
	"encoding/json"
)

// GenerateRequest represents SGLang's native generate request
// Similar to Rust GenerateRequest
type GenerateRequest struct {
	// Text input - SGLang native format
	Text *string `json:"text,omitempty"`

	// Input IDs for tokenized input
	// Can be []uint32 (single) or [][]uint32 (batch)
	InputIDs interface{} `json:"input_ids,omitempty"`

	// Sampling parameters (SGLang style)
	SamplingParams *SamplingParams `json:"sampling_params,omitempty"`

	// Whether to stream the response
	Stream bool `json:"stream,omitempty"`

	// Whether to return logprobs
	ReturnLogprob *bool `json:"return_logprob,omitempty"`

	// If return logprobs, the start location in the prompt
	LogprobStartLen *int32 `json:"logprob_start_len,omitempty"`

	// If return logprobs, the number of top logprobs to return
	TopLogprobsNum *int32 `json:"top_logprobs_num,omitempty"`

	// If return logprobs, the token ids to return logprob for
	TokenIDsLogprob []uint32 `json:"token_ids_logprob,omitempty"`

	// Whether to return model hidden states
	ReturnHiddenStates bool `json:"return_hidden_states,omitempty"`

	// For disaggregated inference
	BootstrapHost *string `json:"bootstrap_host,omitempty"`
	BootstrapPort *int32  `json:"bootstrap_port,omitempty"`
	BootstrapRoom *int32  `json:"bootstrap_room,omitempty"`

	// Data parallel rank routing
	DataParallelRank *int32 `json:"data_parallel_rank,omitempty"`

	// Session parameters for continual prompting
	SessionParams map[string]interface{} `json:"session_params,omitempty"`

	// Path to LoRA adapter
	LoraPath *string `json:"lora_path,omitempty"`
	LoraID   *string `json:"lora_id,omitempty"`

	// Whether to log metrics
	LogMetrics bool `json:"log_metrics,omitempty"`

	// Additional fields
	ExtraFields json.RawMessage `json:"-"` // For extensibility
}

// SamplingParams represents sampling parameters
// Similar to Rust SamplingParams
type SamplingParams struct {
	Temperature       *float32 `json:"temperature,omitempty"`
	TopP              *float32 `json:"top_p,omitempty"`
	TopK              *int32   `json:"top_k,omitempty"`
	MinP              *float32 `json:"min_p,omitempty"`
	FrequencyPenalty  *float32 `json:"frequency_penalty,omitempty"`
	PresencePenalty   *float32 `json:"presence_penalty,omitempty"`
	RepetitionPenalty *float32 `json:"repetition_penalty,omitempty"`
	MaxNewTokens      *int32   `json:"max_new_tokens,omitempty"`
	MinNewTokens      *int32   `json:"min_new_tokens,omitempty"`
	Stop              []string `json:"stop,omitempty"`
	StopTokenIDs      []uint32 `json:"stop_token_ids,omitempty"`
	SkipSpecialTokens *bool    `json:"skip_special_tokens,omitempty"`
	NoStopTrim        *bool    `json:"no_stop_trim,omitempty"`
	IgnoreEOS         *bool    `json:"ignore_eos,omitempty"`
	N                 *int32   `json:"n,omitempty"` // Number of samples

	// Structured generation constraints
	Regex       *string `json:"regex,omitempty"`
	JSONSchema  *string `json:"json_schema,omitempty"`
	EBNFGrammar *string `json:"ebnf_grammar,omitempty"`

	// Logit bias
	LogitBias map[string]float32 `json:"logit_bias,omitempty"`

	// Stream interval
	StreamInterval *int32 `json:"stream_interval,omitempty"`
}

// GenerateResponse represents SGLang's native generate response
// Similar to Rust GenerateResponse
type GenerateResponse struct {
	// Generated text
	Text string `json:"text,omitempty"`

	// Output token IDs
	OutputIDs []uint32 `json:"output_ids,omitempty"`

	// Finish reason
	FinishReason string `json:"finish_reason"` // "stop", "length", "abort"

	// Token usage
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	CachedTokens     int `json:"cached_tokens,omitempty"`

	// Logprobs if requested
	Logprobs interface{} `json:"logprobs,omitempty"`

	// Hidden states if requested
	HiddenStates interface{} `json:"hidden_states,omitempty"`

	// Matched stop information
	MatchedStopTokenID *uint32 `json:"matched_stop_token_id,omitempty"`
	MatchedStopStr     *string `json:"matched_stop_str,omitempty"`

	// Request metadata
	RequestID string `json:"request_id,omitempty"`
}

// IsStreaming returns whether the request is for streaming
func (r *GenerateRequest) IsStreaming() bool {
	return r.Stream
}

// GetModel returns empty string (generate requests don't have model field)
func (r *GenerateRequest) GetModel() string {
	return ""
}
