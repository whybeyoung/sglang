package tokenizer

import (
	"fmt"
	"path/filepath"
	"unsafe"

	"go.uber.org/zap"
)

/*
#cgo CFLAGS: -I${SRCDIR}/../../tokenizer-ffi
#cgo LDFLAGS: -L${SRCDIR}/../../tokenizer-ffi/target/release -ltokenizer_ffi
#include <stdlib.h>

// Forward declarations (will be in header file after build)
typedef struct TokenizerError TokenizerError;

extern TokenizerError* tokenizer_from_file(const char* path);
extern void tokenizer_error_free(TokenizerError* err);
extern int tokenizer_error_code(const TokenizerError* err);
extern const char* tokenizer_error_message(const TokenizerError* err);

extern int tokenizer_vocab_size(void);
extern TokenizerError* tokenizer_encode(const char* text, unsigned int* output, int output_capacity, int* output_len);
extern TokenizerError* tokenizer_decode(const unsigned int* token_ids, int token_ids_len, int skip_special_tokens, char* output, int output_capacity, int* output_len);
extern int tokenizer_token_to_id(const char* token);
*/
import "C"

// RustFFITokenizer implements Tokenizer using Rust tokenizers via CGO
// This provides better compatibility with HuggingFace tokenizers
type RustFFITokenizer struct {
	initialized        bool
	logger             *zap.Logger
	chatTemplate       *string
	chatTemplateFormat ChatTemplateContentFormat
}

// NewRustFFITokenizer creates a new Rust FFI tokenizer
func NewRustFFITokenizer(tokenizerPath string, logger *zap.Logger) (*RustFFITokenizer, error) {
	return NewRustFFITokenizerWithChatTemplate(tokenizerPath, nil, logger)
}

// NewRustFFITokenizerWithChatTemplate creates a Rust FFI tokenizer with optional chat template
func NewRustFFITokenizerWithChatTemplate(tokenizerPath string, chatTemplatePath *string, logger *zap.Logger) (*RustFFITokenizer, error) {
	t := &RustFFITokenizer{
		logger:             logger,
		chatTemplateFormat: ChatTemplateContentFormatString,
	}

	// Initialize tokenizer from file
	cPath := C.CString(tokenizerPath)
	defer C.free(unsafe.Pointer(cPath))

	err := C.tokenizer_from_file(cPath)
	if err == nil {
		return nil, fmt.Errorf("tokenizer_from_file returned null")
	}

	// Check for errors
	code := C.tokenizer_error_code(err)
	if code != 0 {
		msg := C.tokenizer_error_message(err)
		var errMsg string
		if msg != nil {
			errMsg = C.GoString(msg)
		}
		C.tokenizer_error_free(err)
		return nil, fmt.Errorf("failed to load tokenizer (code %d): %s", code, errMsg)
	}

	// Free the error object (success case)
	C.tokenizer_error_free(err)

	// Load chat template if available
	var chatTemplate *string
	var templateFormat ChatTemplateContentFormat

	// Try to discover chat template from model directory
	if chatTemplatePath != nil {
		template, err := LoadChatTemplateFromConfig(*chatTemplatePath)
		if err == nil && template != nil {
			chatTemplate = template
			templateFormat = DetectChatTemplateContentFormat(*template)
		}
	} else {
		// Auto-discover chat template from tokenizer directory
		dir := filepath.Dir(tokenizerPath)
		discovered := discoverChatTemplateInDir(dir)
		if discovered != nil {
			chatTemplate = discovered
			templateFormat = DetectChatTemplateContentFormat(*discovered)
		}
	}

	// Store chat template (if available)
	t.chatTemplate = chatTemplate
	t.chatTemplateFormat = templateFormat
	if chatTemplate != nil {
		logger.Info("Chat template loaded for Rust FFI tokenizer",
			zap.String("path", tokenizerPath),
			zap.String("format", string(templateFormat)),
		)
	}

	t.initialized = true
	logger.Info("Rust FFI tokenizer initialized",
		zap.String("path", tokenizerPath),
		zap.Bool("has_chat_template", chatTemplate != nil),
	)

	return t, nil
}

// Encode encodes text into token IDs
func (t *RustFFITokenizer) Encode(text string) (*Encoding, error) {
	if !t.initialized {
		return nil, fmt.Errorf("tokenizer not initialized")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	// Allocate buffer for token IDs (reasonable max size)
	const maxTokens = 100000
	output := make([]C.uint, maxTokens)
	var outputLen C.int

	err := C.tokenizer_encode(
		cText,
		(*C.uint)(unsafe.Pointer(&output[0])),
		maxTokens,
		&outputLen,
	)

	if err == nil {
		return nil, fmt.Errorf("tokenizer_encode returned null")
	}

	code := C.tokenizer_error_code(err)
	if code != 0 {
		msg := C.tokenizer_error_message(err)
		var errMsg string
		if msg != nil {
			errMsg = C.GoString(msg)
		}
		C.tokenizer_error_free(err)
		return nil, fmt.Errorf("encoding failed (code %d): %s", code, errMsg)
	}

	C.tokenizer_error_free(err)

	// Convert to Go slice
	tokenIDs := make([]uint32, int(outputLen))
	for i := 0; i < int(outputLen); i++ {
		tokenIDs[i] = uint32(output[i])
	}

	return &Encoding{
		TokenIDs: tokenIDs,
		Tokens:   nil, // Tokens not available from Rust FFI yet
	}, nil
}

// Decode decodes token IDs to text
func (t *RustFFITokenizer) Decode(tokenIDs []uint32, skipSpecialTokens bool) (string, error) {
	if !t.initialized {
		return "", fmt.Errorf("tokenizer not initialized")
	}

	if len(tokenIDs) == 0 {
		return "", nil
	}

	// Allocate buffer for decoded text
	const maxOutputSize = 1024 * 1024 // 1MB max
	output := make([]byte, maxOutputSize)
	var outputLen C.int

	var skipSpecial C.int
	if skipSpecialTokens {
		skipSpecial = 1
	}

	err := C.tokenizer_decode(
		(*C.uint)(unsafe.Pointer(&tokenIDs[0])),
		C.int(len(tokenIDs)),
		skipSpecial,
		(*C.char)(unsafe.Pointer(&output[0])),
		maxOutputSize,
		&outputLen,
	)

	if err == nil {
		return "", fmt.Errorf("tokenizer_decode returned null")
	}

	code := C.tokenizer_error_code(err)
	if code != 0 {
		msg := C.tokenizer_error_message(err)
		var errMsg string
		if msg != nil {
			errMsg = C.GoString(msg)
		}
		C.tokenizer_error_free(err)
		return "", fmt.Errorf("decoding failed (code %d): %s", code, errMsg)
	}

	C.tokenizer_error_free(err)

	if outputLen <= 0 {
		return "", nil
	}

	// Convert to Go string (null-terminated C string)
	result := C.GoString((*C.char)(unsafe.Pointer(&output[0])))
	return result, nil
}

// GetVocabSize returns vocabulary size
func (t *RustFFITokenizer) GetVocabSize() int {
	if !t.initialized {
		return 0
	}

	size := C.tokenizer_vocab_size()
	if size < 0 {
		return 0
	}

	// size == 0 means "available but size unknown"
	// Return a placeholder value
	return 50000 // Placeholder
}

// GetEOSTokenID returns EOS token ID
func (t *RustFFITokenizer) GetEOSTokenID() uint32 {
	// TODO: Implement when FFI supports special tokens
	return 2 // Placeholder
}

// GetPADTokenID returns PAD token ID
func (t *RustFFITokenizer) GetPADTokenID() uint32 {
	// TODO: Implement when FFI supports special tokens
	return 0 // Placeholder
}

// TokenToId returns token ID for a token string
func (t *RustFFITokenizer) TokenToId(token string) (uint32, bool) {
	if !t.initialized {
		return 0, false
	}

	cToken := C.CString(token)
	defer C.free(unsafe.Pointer(cToken))

	id := C.tokenizer_token_to_id(cToken)
	if id < 0 {
		return 0, false
	}

	return uint32(id), true
}

// ChatTemplate returns chat template if available
func (t *RustFFITokenizer) ChatTemplate() *string {
	return t.chatTemplate
}

// ChatTemplateContentFormat returns chat template content format
func (t *RustFFITokenizer) ChatTemplateContentFormat() ChatTemplateContentFormat {
	return t.chatTemplateFormat
}
