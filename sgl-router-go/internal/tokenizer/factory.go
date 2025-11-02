package tokenizer

import (
	"fmt"
	"os"
	"path/filepath"

	"go.uber.org/zap"
)

// CreateTokenizerFromFile creates a tokenizer from a file path
// Similar to Rust create_tokenizer_from_file
func CreateTokenizerFromFile(filePath string, logger *zap.Logger) (Tokenizer, error) {
	return CreateTokenizerWithChatTemplate(filePath, nil, logger)
}

// CreateTokenizerWithChatTemplate creates a tokenizer with optional chat template
// Similar to Rust create_tokenizer_with_chat_template
func CreateTokenizerWithChatTemplate(
	filePath string,
	chatTemplatePath *string,
	logger *zap.Logger,
) (Tokenizer, error) {
	// Special case for testing
	if filePath == "mock" || filePath == "test" {
		return NewMockTokenizer(), nil
	}

	path := filepath.Clean(filePath)

	// Check if file exists
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, fmt.Errorf("tokenizer file not found: %s", filePath)
	}

	// If path is a directory, search for tokenizer files
	info, err := os.Stat(path)
	if err == nil && info.IsDir() {
		tokenizerJSON := filepath.Join(path, "tokenizer.json")
		if _, err := os.Stat(tokenizerJSON); err == nil {
			// Resolve chat template: provided path takes precedence over auto-discovery
			var finalChatTemplate *string
			if chatTemplatePath != nil {
				finalChatTemplate = chatTemplatePath
			} else {
				// Try to auto-discover chat template
				discovered := discoverChatTemplateInDir(path)
				finalChatTemplate = discovered
			}

			return NewHuggingFaceTokenizerWithChatTemplate(tokenizerJSON, finalChatTemplate, logger)
		}

		return nil, fmt.Errorf(
			"directory '%s' does not contain a valid tokenizer file (tokenizer.json)",
			filePath,
		)
	}

	// Try to determine tokenizer type from extension
	ext := filepath.Ext(path)
	switch ext {
	case ".json":
		// HuggingFace tokenizer JSON file
		// Try Rust FFI tokenizer first (better compatibility)
		// Fallback to Go tokenizer if Rust FFI is not available
		if rustTokenizer, err := NewRustFFITokenizerWithChatTemplate(path, chatTemplatePath, logger); err == nil {
			logger.Info("Using Rust FFI tokenizer",
				zap.String("path", path),
			)
			return rustTokenizer, nil
		} else {
			logger.Debug("Rust FFI tokenizer not available, using Go tokenizer",
				zap.String("path", path),
				zap.Error(err),
			)
		}
		return NewHuggingFaceTokenizerWithChatTemplate(path, chatTemplatePath, logger)
	default:
		// Try as HuggingFace tokenizer anyway
		// Try Rust FFI first
		if rustTokenizer, err := NewRustFFITokenizerWithChatTemplate(path, chatTemplatePath, logger); err == nil {
			logger.Info("Using Rust FFI tokenizer",
				zap.String("path", path),
			)
			return rustTokenizer, nil
		}
		return NewHuggingFaceTokenizerWithChatTemplate(path, chatTemplatePath, logger)
	}
}

// discoverChatTemplateInDir tries to discover chat template in a directory
// Similar to Rust discover_chat_template_in_dir
func discoverChatTemplateInDir(dir string) *string {
	candidates := []string{
		"tokenizer_config.json",
		"chat_template.json",
		"chat_template.jinja",
	}

	for _, candidate := range candidates {
		path := filepath.Join(dir, candidate)
		if _, err := os.Stat(path); err == nil {
			template, err := loadChatTemplateFromFile(path)
			if err == nil && template != nil {
				return template
			}
		}
	}

	return nil
}

// CreateTokenizerWithChatTemplateBlocking creates a tokenizer (blocking version)
// Similar to Rust create_tokenizer_with_chat_template_blocking
// For now, this is the same as CreateTokenizerWithChatTemplate
// (In Rust, this handles async downloads from HuggingFace Hub)
func CreateTokenizerWithChatTemplateBlocking(
	modelNameOrPath string,
	chatTemplatePath *string,
	logger *zap.Logger,
) (Tokenizer, error) {
	// Check if it's a file path
	path := filepath.Clean(modelNameOrPath)
	if _, err := os.Stat(path); err == nil {
		return CreateTokenizerWithChatTemplate(path, chatTemplatePath, logger)
	}

	// TODO: Support downloading from HuggingFace Hub
	// For now, return error if path doesn't exist
	return nil, fmt.Errorf(
		"tokenizer path not found: %s (HuggingFace Hub download not yet implemented)",
		modelNameOrPath,
	)
}
