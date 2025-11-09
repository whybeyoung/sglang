// Package ffi provides Go bindings for sgl-router FFI functions
package ffi

/*
#cgo LDFLAGS: -lsglang_router_rs -ldl
#include <stdlib.h>
#include <stdint.h>

// Error codes
typedef enum {
    SGL_ERROR_SUCCESS = 0,
    SGL_ERROR_INVALID_ARGUMENT = 1,
    SGL_ERROR_TOKENIZATION_ERROR = 2,
    SGL_ERROR_PARSING_ERROR = 3,
    SGL_ERROR_MEMORY_ERROR = 4,
    SGL_ERROR_UNKNOWN = 99
} SglErrorCode;

// Opaque handles
typedef struct TokenizerHandle TokenizerHandle;

// Tokenizer functions
TokenizerHandle* sgl_tokenizer_create_from_file(const char* path, char** error_out);
SglErrorCode sgl_tokenizer_encode(
    TokenizerHandle* handle,
    const char* text,
    uint32_t** token_ids_out,
    size_t* token_count_out,
    char** error_out
);
SglErrorCode sgl_tokenizer_apply_chat_template(
    TokenizerHandle* handle,
    const char* messages_json,
    char** result_out,
    char** error_out
);
SglErrorCode sgl_tokenizer_apply_chat_template_with_tools(
    TokenizerHandle* handle,
    const char* messages_json,
    const char* tools_json,
    char** result_out,
    char** error_out
);
SglErrorCode sgl_tokenizer_decode(
    TokenizerHandle* handle,
    const uint32_t* token_ids,
    size_t token_count,
    int skip_special_tokens,
    char** result_out,
    char** error_out
);
void sgl_tokenizer_free(TokenizerHandle* handle);

// Memory management
void sgl_free_string(char* str);
void sgl_free_token_ids(uint32_t* ptr, size_t count);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// ErrorCode represents FFI error codes
type ErrorCode int

const (
	ErrorSuccess           ErrorCode = 0
	ErrorInvalidArgument   ErrorCode = 1
	ErrorTokenizationError ErrorCode = 2
	ErrorParsingError      ErrorCode = 3
	ErrorMemoryError       ErrorCode = 4
	ErrorUnknown           ErrorCode = 99
)

func (e ErrorCode) Error() string {
	switch e {
	case ErrorSuccess:
		return "success"
	case ErrorInvalidArgument:
		return "invalid argument"
	case ErrorTokenizationError:
		return "tokenization error"
	case ErrorParsingError:
		return "parsing error"
	case ErrorMemoryError:
		return "memory error"
	case ErrorUnknown:
		return "unknown error"
	default:
		return fmt.Sprintf("unknown error code: %d", e)
	}
}

// TokenizerHandle wraps the Rust tokenizer FFI handle
type TokenizerHandle struct {
	handle *C.TokenizerHandle
}

// NewTokenizerFromFile creates a tokenizer from a file path
func NewTokenizerFromFile(path string) (*TokenizerHandle, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var errorOut *C.char
	handle := C.sgl_tokenizer_create_from_file(cPath, &errorOut)

	if handle == nil {
		if errorOut != nil {
			errMsg := C.GoString(errorOut)
			C.sgl_free_string(errorOut)
			return nil, fmt.Errorf("failed to create tokenizer: %s", errMsg)
		}
		return nil, fmt.Errorf("failed to create tokenizer: unknown error")
	}

	return &TokenizerHandle{handle: handle}, nil
}

// Encode encodes text to token IDs
func (t *TokenizerHandle) Encode(text string) ([]uint32, error) {
	if t.handle == nil {
		return nil, fmt.Errorf("tokenizer handle is nil")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var tokenIds *C.uint32_t
	var tokenCount C.size_t
	var errorOut *C.char

	result := C.sgl_tokenizer_encode(t.handle, cText, &tokenIds, &tokenCount, &errorOut)

	if result != C.SGL_ERROR_SUCCESS {
		if errorOut != nil {
			errMsg := C.GoString(errorOut)
			C.sgl_free_string(errorOut)
			return nil, fmt.Errorf("encode failed: %s", errMsg)
		}
		return nil, fmt.Errorf("encode failed: error code %d", result)
	}

	if tokenCount == 0 {
		return []uint32{}, nil
	}

	// Convert C array to Go slice
	tokens := (*[1 << 28]uint32)(unsafe.Pointer(tokenIds))[:tokenCount:tokenCount]
	resultTokens := make([]uint32, tokenCount)
	copy(resultTokens, tokens)

	// Free the C-allocated memory
	C.sgl_free_token_ids(tokenIds, tokenCount)

	return resultTokens, nil
}

// ApplyChatTemplate applies chat template to messages
func (t *TokenizerHandle) ApplyChatTemplate(messagesJSON string) (string, error) {
	if t.handle == nil {
		return "", fmt.Errorf("tokenizer handle is nil")
	}

	cMessages := C.CString(messagesJSON)
	defer C.free(unsafe.Pointer(cMessages))

	var resultOut *C.char
	var errorOut *C.char

	result := C.sgl_tokenizer_apply_chat_template(t.handle, cMessages, &resultOut, &errorOut)

	if result != C.SGL_ERROR_SUCCESS {
		if errorOut != nil {
			errMsg := C.GoString(errorOut)
			C.sgl_free_string(errorOut)
			return "", fmt.Errorf("apply chat template failed: %s", errMsg)
		}
		return "", fmt.Errorf("apply chat template failed: error code %d", result)
	}

	if resultOut == nil {
		return "", fmt.Errorf("result is nil")
	}

	resultStr := C.GoString(resultOut)
	C.sgl_free_string(resultOut)

	return resultStr, nil
}

// ApplyChatTemplateWithTools applies chat template to messages with tools support
func (t *TokenizerHandle) ApplyChatTemplateWithTools(messagesJSON, toolsJSON string) (string, error) {
	if t.handle == nil {
		return "", fmt.Errorf("tokenizer handle is nil")
	}

	cMessages := C.CString(messagesJSON)
	defer C.free(unsafe.Pointer(cMessages))

	var cTools *C.char
	if toolsJSON != "" {
		cTools = C.CString(toolsJSON)
		defer C.free(unsafe.Pointer(cTools))
	}

	var resultOut *C.char
	var errorOut *C.char

	result := C.sgl_tokenizer_apply_chat_template_with_tools(t.handle, cMessages, cTools, &resultOut, &errorOut)

	if result != C.SGL_ERROR_SUCCESS {
		if errorOut != nil {
			errMsg := C.GoString(errorOut)
			C.sgl_free_string(errorOut)
			return "", fmt.Errorf("apply chat template failed: %s", errMsg)
		}
		return "", fmt.Errorf("apply chat template failed: error code %d", result)
	}

	if resultOut == nil {
		return "", fmt.Errorf("result is nil")
	}

	resultStr := C.GoString(resultOut)
	C.sgl_free_string(resultOut)

	return resultStr, nil
}

// Decode decodes token IDs to text
func (t *TokenizerHandle) Decode(tokenIds []uint32, skipSpecialTokens bool) (string, error) {
	if t.handle == nil {
		return "", fmt.Errorf("tokenizer handle is nil")
	}

	if len(tokenIds) == 0 {
		return "", nil
	}

	var resultOut *C.char
	var errorOut *C.char
	skipSpecial := C.int(0)
	if skipSpecialTokens {
		skipSpecial = C.int(1)
	}

	result := C.sgl_tokenizer_decode(
		t.handle,
		(*C.uint32_t)(&tokenIds[0]),
		C.size_t(len(tokenIds)),
		skipSpecial,
		&resultOut,
		&errorOut,
	)

	if result != C.SGL_ERROR_SUCCESS {
		if errorOut != nil {
			errMsg := C.GoString(errorOut)
			C.sgl_free_string(errorOut)
			return "", fmt.Errorf("decode failed: %s", errMsg)
		}
		return "", fmt.Errorf("decode failed: error code %d", result)
	}

	if resultOut == nil {
		return "", fmt.Errorf("result is nil")
	}

	resultStr := C.GoString(resultOut)
	C.sgl_free_string(resultOut)

	return resultStr, nil
}

// Close frees the tokenizer handle
func (t *TokenizerHandle) Close() error {
	if t.handle != nil {
		C.sgl_tokenizer_free(t.handle)
		t.handle = nil
	}
	return nil
}
