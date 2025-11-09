// Package ffi provides Go bindings for sgl-router tool parser FFI functions
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
typedef struct ToolParserHandle ToolParserHandle;

// Memory management (must match tokenizer.go)
void sgl_free_string(char* str);

// Tool parser functions
ToolParserHandle* sgl_tool_parser_create(const char* parser_type, char** error_out);
SglErrorCode sgl_tool_parser_parse_complete(
    ToolParserHandle* handle,
    const char* text,
    char** result_json_out,
    char** error_out
);
SglErrorCode sgl_tool_parser_parse_incremental(
    ToolParserHandle* handle,
    const char* chunk,
    const char* tools_json,
    char** result_json_out,
    char** error_out
);
void sgl_tool_parser_reset(ToolParserHandle* handle);
void sgl_tool_parser_free(ToolParserHandle* handle);
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"unsafe"
)

// ToolParserHandle wraps the Rust tool parser FFI handle
type ToolParserHandle struct {
	handle *C.ToolParserHandle
}

// ToolParserResult represents the result of parsing tool calls
type ToolParserResult struct {
	NormalText string     `json:"normal_text"`
	ToolCalls  []ToolCall `json:"tool_calls"`
}

// ToolCall represents a parsed tool call in OpenAI format
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction represents the function part of a tool call
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

// NewToolParser creates a new tool parser
func NewToolParser(parserType string) (*ToolParserHandle, error) {
	cParserType := C.CString(parserType)
	defer C.free(unsafe.Pointer(cParserType))

	var errorOut *C.char
	handle := C.sgl_tool_parser_create(cParserType, &errorOut)

	if handle == nil {
		if errorOut != nil {
			errMsg := C.GoString(errorOut)
			C.sgl_free_string(errorOut)
			return nil, fmt.Errorf("failed to create tool parser: %s", errMsg)
		}
		return nil, fmt.Errorf("failed to create tool parser: unknown error")
	}

	return &ToolParserHandle{handle: handle}, nil
}

// ParseComplete parses complete tool calls from text
func (h *ToolParserHandle) ParseComplete(text string) (*ToolParserResult, error) {
	if h.handle == nil {
		return nil, fmt.Errorf("tool parser handle is nil")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var resultOut *C.char
	var errorOut *C.char

	code := C.sgl_tool_parser_parse_complete(h.handle, cText, &resultOut, &errorOut)

	if code != C.SGL_ERROR_SUCCESS {
		if errorOut != nil {
			errMsg := C.GoString(errorOut)
			C.sgl_free_string(errorOut)
			return nil, fmt.Errorf("parse error: %s", errMsg)
		}
		return nil, fmt.Errorf("parse error: unknown error")
	}

	if resultOut == nil {
		return nil, fmt.Errorf("parse error: result is nil")
	}

	resultJSON := C.GoString(resultOut)
	C.sgl_free_string(resultOut)

	var result ToolParserResult
	if err := json.Unmarshal([]byte(resultJSON), &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal result: %w", err)
	}

	return &result, nil
}

// ParseIncremental parses tool calls incrementally from streaming chunks
func (h *ToolParserHandle) ParseIncremental(chunk string, toolsJSON string) (*ToolParserResult, error) {
	if h.handle == nil {
		return nil, fmt.Errorf("tool parser handle is nil")
	}

	cChunk := C.CString(chunk)
	defer C.free(unsafe.Pointer(cChunk))

	var cToolsJSON *C.char
	if toolsJSON != "" {
		cToolsJSON = C.CString(toolsJSON)
		defer C.free(unsafe.Pointer(cToolsJSON))
	}

	var resultOut *C.char
	var errorOut *C.char

	code := C.sgl_tool_parser_parse_incremental(h.handle, cChunk, cToolsJSON, &resultOut, &errorOut)

	if code != C.SGL_ERROR_SUCCESS {
		if errorOut != nil {
			errMsg := C.GoString(errorOut)
			C.sgl_free_string(errorOut)
			return nil, fmt.Errorf("parse incremental error: %s", errMsg)
		}
		return nil, fmt.Errorf("parse incremental error: unknown error")
	}

	if resultOut == nil {
		return nil, fmt.Errorf("parse incremental error: result is nil")
	}

	resultJSON := C.GoString(resultOut)
	C.sgl_free_string(resultOut)

	var result ToolParserResult
	if err := json.Unmarshal([]byte(resultJSON), &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal result: %w", err)
	}

	return &result, nil
}

// Reset resets the parser state for reuse
func (h *ToolParserHandle) Reset() {
	if h.handle != nil {
		C.sgl_tool_parser_reset(h.handle)
	}
}

// Close frees the tool parser handle
func (h *ToolParserHandle) Close() error {
	if h.handle != nil {
		C.sgl_tool_parser_free(h.handle)
		h.handle = nil
	}
	return nil
}
