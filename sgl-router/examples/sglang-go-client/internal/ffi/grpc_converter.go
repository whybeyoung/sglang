// Package ffi provides Go bindings for sgl-router gRPC response converter FFI functions
package ffi

/*
#cgo LDFLAGS: -lsglang_router_rs -ldl
#include <stdlib.h>
#include <stdint.h>

// Error codes (must match tokenizer.go)
typedef enum {
    SGL_ERROR_SUCCESS = 0,
    SGL_ERROR_INVALID_ARGUMENT = 1,
    SGL_ERROR_TOKENIZATION_ERROR = 2,
    SGL_ERROR_PARSING_ERROR = 3,
    SGL_ERROR_MEMORY_ERROR = 4,
    SGL_ERROR_UNKNOWN = 99
} SglErrorCode;

// Opaque handles (forward declarations)
typedef struct TokenizerHandle TokenizerHandle;
typedef struct GrpcResponseConverterHandle GrpcResponseConverterHandle;

// Memory management (must match tokenizer.go)
void sgl_free_string(char* str);

// gRPC response converter functions
GrpcResponseConverterHandle* sgl_grpc_response_converter_create(
    TokenizerHandle* tokenizer_handle,
    const char* model,
    const char* request_id,
    const char* tools_json,
    const char* tool_choice_json,
    const char* stop,
    const char* stop_token_ids,
    int skip_special_tokens,
    char** error_out
);
SglErrorCode sgl_grpc_response_converter_convert_chunk(
    GrpcResponseConverterHandle* handle,
    const char* response_json,
    char** result_json_out,
    char** error_out
);
void sgl_grpc_response_converter_free(GrpcResponseConverterHandle* handle);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// GrpcResponseConverterHandle wraps the Rust gRPC response converter FFI handle
type GrpcResponseConverterHandle struct {
	handle *C.GrpcResponseConverterHandle
}

// NewGrpcResponseConverter creates a new gRPC response converter
func NewGrpcResponseConverter(
	tokenizerHandle *TokenizerHandle,
	model string,
	requestID string,
	toolsJSON string,
	toolChoiceJSON string,
	stop string,
	stopTokenIDs string,
	skipSpecialTokens bool,
) (*GrpcResponseConverterHandle, error) {
	if tokenizerHandle == nil || tokenizerHandle.handle == nil {
		return nil, fmt.Errorf("tokenizer handle is nil")
	}

	cModel := C.CString(model)
	defer C.free(unsafe.Pointer(cModel))

	cRequestID := C.CString(requestID)
	defer C.free(unsafe.Pointer(cRequestID))

	var cToolsJSON *C.char
	if toolsJSON != "" {
		cToolsJSON = C.CString(toolsJSON)
		defer C.free(unsafe.Pointer(cToolsJSON))
	}

	var cToolChoiceJSON *C.char
	if toolChoiceJSON != "" {
		cToolChoiceJSON = C.CString(toolChoiceJSON)
		defer C.free(unsafe.Pointer(cToolChoiceJSON))
	}

	var cStop *C.char
	if stop != "" {
		cStop = C.CString(stop)
		defer C.free(unsafe.Pointer(cStop))
	}

	var cStopTokenIDs *C.char
	if stopTokenIDs != "" {
		cStopTokenIDs = C.CString(stopTokenIDs)
		defer C.free(unsafe.Pointer(cStopTokenIDs))
	}

	var errorOut *C.char
	skipSpecial := C.int(0)
	if skipSpecialTokens {
		skipSpecial = C.int(1)
	}

	handle := C.sgl_grpc_response_converter_create(
		tokenizerHandle.handle,
		cModel,
		cRequestID,
		cToolsJSON,
		cToolChoiceJSON,
		cStop,
		cStopTokenIDs,
		skipSpecial,
		&errorOut,
	)

	if handle == nil {
		if errorOut != nil {
			errMsg := C.GoString(errorOut)
			C.sgl_free_string(errorOut)
			return nil, fmt.Errorf("failed to create gRPC response converter: %s", errMsg)
		}
		return nil, fmt.Errorf("failed to create gRPC response converter: unknown error")
	}

	return &GrpcResponseConverterHandle{handle: handle}, nil
}

// ConvertChunk converts a gRPC GenerateResponse chunk to OpenAI format
func (h *GrpcResponseConverterHandle) ConvertChunk(responseJSON string) (string, error) {
	if h.handle == nil {
		return "", fmt.Errorf("converter handle is nil")
	}

	cResponseJSON := C.CString(responseJSON)
	defer C.free(unsafe.Pointer(cResponseJSON))

	var resultOut *C.char
	var errorOut *C.char

	code := C.sgl_grpc_response_converter_convert_chunk(h.handle, cResponseJSON, &resultOut, &errorOut)

	if code != C.SGL_ERROR_SUCCESS {
		if errorOut != nil {
			errMsg := C.GoString(errorOut)
			C.sgl_free_string(errorOut)
			return "", fmt.Errorf("convert chunk error: %s", errMsg)
		}
		return "", fmt.Errorf("convert chunk error: unknown error")
	}

	if resultOut == nil {
		return "", nil // Empty response (e.g., empty chunk)
	}

	resultJSON := C.GoString(resultOut)
	C.sgl_free_string(resultOut)

	return resultJSON, nil
}

// Close frees the gRPC response converter handle
func (h *GrpcResponseConverterHandle) Close() error {
	if h.handle != nil {
		C.sgl_grpc_response_converter_free(h.handle)
		h.handle = nil
	}
	return nil
}

// ChatCompletionStreamResponse represents OpenAI streaming response format
type ChatCompletionStreamResponse struct {
	ID                string             `json:"id"`
	Object            string             `json:"object"`
	Created           int64              `json:"created"`
	Model             string             `json:"model"`
	SystemFingerprint *string            `json:"system_fingerprint,omitempty"`
	Choices           []ChatStreamChoice `json:"choices"`
	Usage             *Usage             `json:"usage,omitempty"`
}

// ChatStreamChoice represents a choice in streaming response
type ChatStreamChoice struct {
	Index        int32            `json:"index"`
	Delta        ChatMessageDelta `json:"delta"`
	Logprobs     interface{}      `json:"logprobs,omitempty"`
	FinishReason *string          `json:"finish_reason,omitempty"`
	MatchedStop  interface{}      `json:"matched_stop,omitempty"`
}

// ChatMessageDelta represents the delta in streaming response
type ChatMessageDelta struct {
	Role             *string         `json:"role,omitempty"`
	Content          *string         `json:"content,omitempty"`
	ToolCalls        []ToolCallDelta `json:"tool_calls,omitempty"`
	ReasoningContent *string         `json:"reasoning_content,omitempty"`
}

// ToolCallDelta represents a tool call delta
type ToolCallDelta struct {
	Index    int32              `json:"index"`
	ID       *string            `json:"id,omitempty"`
	Type     *string            `json:"type,omitempty"`
	Function *FunctionCallDelta `json:"function,omitempty"`
}

// FunctionCallDelta represents a function call delta
type FunctionCallDelta struct {
	Name      *string `json:"name,omitempty"`
	Arguments *string `json:"arguments,omitempty"`
}

// Usage represents token usage
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
