package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	"go.uber.org/zap"
)

// Note: Import tokenizer package for EncodeInput types

// HuggingFaceTokenizer implements Tokenizer using the sugarme/tokenizer library
// Similar to Rust HuggingFaceTokenizer
type HuggingFaceTokenizer struct {
	tokenizer *tokenizer.Tokenizer
	vocab     map[string]uint32
	// reverseVocab maps token ID -> token string
	reverseVocab       map[uint32]string
	specialTokens      SpecialTokens
	chatTemplate       *string
	chatTemplateFormat ChatTemplateContentFormat
	vocabSize          int
	logger             *zap.Logger
}

// SpecialTokens contains special token information
type SpecialTokens struct {
	BOSToken                *string
	EOSToken                *string
	UNKToken                *string
	SEPToken                *string
	PADToken                *string
	CLSToken                *string
	MaskToken               *string
	AdditionalSpecialTokens []string
}

// NewHuggingFaceTokenizer creates a new HuggingFace tokenizer from a file path
// Similar to Rust HuggingFaceTokenizer::from_file
func NewHuggingFaceTokenizer(filePath string, logger *zap.Logger) (*HuggingFaceTokenizer, error) {
	return NewHuggingFaceTokenizerWithChatTemplate(filePath, nil, logger)
}

// NewHuggingFaceTokenizerWithChatTemplate creates a tokenizer with optional chat template
// Similar to Rust HuggingFaceTokenizer::from_file_with_chat_template
func NewHuggingFaceTokenizerWithChatTemplate(
	filePath string,
	chatTemplatePath *string,
	logger *zap.Logger,
) (*HuggingFaceTokenizer, error) {
	// Check if file exists
	path := filepath.Clean(filePath)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, fmt.Errorf("tokenizer file not found: %s", filePath)
	}

	// If path is a directory, look for tokenizer.json
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		tokenizerJSON := filepath.Join(path, "tokenizer.json")
		if _, err := os.Stat(tokenizerJSON); err == nil {
			path = tokenizerJSON
		} else {
			return nil, fmt.Errorf("directory '%s' does not contain tokenizer.json", filePath)
		}
	}

	// Load tokenizer from file
	// The sugarme/tokenizer library supports loading from tokenizer.json
	tk, err := pretrained.FromFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer from %s: %w", path, err)
	}

	// Extract special tokens and vocab
	specialTokens := extractSpecialTokens(tk)

	// Build vocab mappings
	vocab := make(map[string]uint32)
	reverseVocab := make(map[uint32]string)

	// Get vocab size
	vocabSize := tk.GetVocabSize(true) // Include special tokens

	// Note: sugarme/tokenizer doesn't expose full vocab API like Rust version
	// We'll build vocab as we use it, or extract from tokenizer if possible
	// For now, we'll use the tokenizer's built-in methods for lookups

	// Load chat template if provided
	var chatTemplate *string
	var chatTemplateSource string // Track where chat template was loaded from
	if chatTemplatePath != nil {
		chatTemplateSource = *chatTemplatePath
		template, err := loadChatTemplateFromFile(*chatTemplatePath)
		if err != nil {
			logger.Warn("Failed to load chat template",
				zap.String("source_file", *chatTemplatePath),
				zap.Error(err),
			)
		} else if template != nil {
			chatTemplate = template
			logger.Info("Chat template loaded from file for HuggingFace tokenizer",
				zap.String("source_file", *chatTemplatePath),
				zap.String("tokenizer_path", path),
			)
		}
	} else {
		// Try to auto-discover chat template from tokenizer_config.json
		template := discoverChatTemplate(path)
		if template != nil {
			chatTemplate = template
			chatTemplateSource = *template
			logger.Info("Chat template auto-discovered from tokenizer directory for HuggingFace tokenizer",
				zap.String("source_file", *template),
				zap.String("tokenizer_path", path),
			)
		}
	}

	// Detect content format if chat template is available
	var templateFormat ChatTemplateContentFormat
	if chatTemplate != nil {
		templateFormat = DetectChatTemplateContentFormat(*chatTemplate)
		logger.Info("Chat template loaded for HuggingFace tokenizer",
			zap.String("tokenizer_path", path),
			zap.String("chat_template_source", chatTemplateSource),
			zap.String("format", templateFormat.String()),
		)
	}

	return &HuggingFaceTokenizer{
		tokenizer:          tk,
		vocab:              vocab,
		reverseVocab:       reverseVocab,
		specialTokens:      specialTokens,
		chatTemplate:       chatTemplate,
		chatTemplateFormat: templateFormat,
		vocabSize:          vocabSize,
		logger:             logger,
	}, nil
}

// ChatTemplate returns the chat template if available
func (h *HuggingFaceTokenizer) ChatTemplate() *string {
	return h.chatTemplate
}

// ChatTemplateContentFormat returns the detected content format
func (h *HuggingFaceTokenizer) ChatTemplateContentFormat() ChatTemplateContentFormat {
	return h.chatTemplateFormat
}

// Encode implements Tokenizer.Encode
func (h *HuggingFaceTokenizer) Encode(text string) (*Encoding, error) {
	// Create EncodeInput from text
	input := tokenizer.NewSingleEncodeInput(tokenizer.NewInputSequence(text))

	// Use the tokenizer's Encode method
	encoding, err := h.tokenizer.Encode(input, false)
	if err != nil {
		return nil, fmt.Errorf("encoding failed: %w", err)
	}

	// Convert to our Encoding type
	ids := encoding.Ids
	tokenIDs := make([]uint32, len(ids))
	tokens := make([]string, len(ids))

	for i, id := range ids {
		tokenIDs[i] = uint32(id)
		// Get token string (if available)
		if len(encoding.Tokens) > i {
			tokens[i] = encoding.Tokens[i]
		} else {
			tokens[i] = fmt.Sprintf("<token_%d>", id)
		}
	}

	return &Encoding{
		TokenIDs: tokenIDs,
		Tokens:   tokens,
	}, nil
}

// Decode implements Tokenizer.Decode
func (h *HuggingFaceTokenizer) Decode(tokenIDs []uint32, skipSpecialTokens bool) (string, error) {
	// Convert uint32 to int (sugarme/tokenizer uses int)
	ids := make([]int, len(tokenIDs))
	for i, id := range tokenIDs {
		ids[i] = int(id)
	}

	// Decode using tokenizer (returns string directly, no error)
	decoded := h.tokenizer.Decode(ids, skipSpecialTokens)

	return decoded, nil
}

// GetVocabSize implements Tokenizer.GetVocabSize
func (h *HuggingFaceTokenizer) GetVocabSize() int {
	return h.vocabSize
}

// GetEOSTokenID returns the EOS token ID
func (h *HuggingFaceTokenizer) GetEOSTokenID() uint32 {
	if h.specialTokens.EOSToken != nil {
		// Try to find the token ID
		id, ok := h.tokenizer.TokenToId(*h.specialTokens.EOSToken)
		if ok && id >= 0 {
			return uint32(id)
		}
	}
	// Default EOS token ID (common in many tokenizers)
	return 2
}

// GetPADTokenID returns the PAD token ID
func (h *HuggingFaceTokenizer) GetPADTokenID() uint32 {
	if h.specialTokens.PADToken != nil {
		// Try to find the token ID
		id, ok := h.tokenizer.TokenToId(*h.specialTokens.PADToken)
		if ok && id >= 0 {
			return uint32(id)
		}
	}
	// Default PAD token ID
	return 0
}

// extractSpecialTokens extracts special tokens from the tokenizer
func extractSpecialTokens(tk *tokenizer.Tokenizer) SpecialTokens {
	special := SpecialTokens{
		AdditionalSpecialTokens: []string{},
	}

	// Try common special token patterns
	patterns := map[string][]string{
		"bos":  []string{"<s>", "<|startoftext|>", "<BOS>", "[CLS]"},
		"eos":  []string{"</s>", "<|endoftext|>", "<EOS>", "[SEP]"},
		"unk":  []string{"<unk>", "<UNK>", "[UNK]"},
		"sep":  []string{"[SEP]", "<sep>", "<SEP]"},
		"pad":  []string{"<pad>", "<PAD>", "[PAD]"},
		"cls":  []string{"[CLS]", "<cls>", "<CLS]"},
		"mask": []string{"[MASK]", "<mask>", "<MASK]"},
	}

	findToken := func(patterns []string) *string {
		for _, pattern := range patterns {
			id, ok := tk.TokenToId(pattern)
			if ok && id >= 0 {
				return &pattern
			}
		}
		return nil
	}

	special.BOSToken = findToken(patterns["bos"])
	special.EOSToken = findToken(patterns["eos"])
	special.UNKToken = findToken(patterns["unk"])
	special.SEPToken = findToken(patterns["sep"])
	special.PADToken = findToken(patterns["pad"])
	special.CLSToken = findToken(patterns["cls"])
	special.MaskToken = findToken(patterns["mask"])

	// Note: Additional special tokens extraction would require more tokenizer introspection
	// which may not be fully available in sugarme/tokenizer

	return special
}

// loadChatTemplateFromFile loads a chat template from a file
func loadChatTemplateFromFile(templatePath string) (*string, error) {
	data, err := os.ReadFile(templatePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read chat template file: %w", err)
	}

	content := strings.TrimSpace(string(data))

	// Check if it's a JSON file containing a Jinja template
	if strings.HasSuffix(templatePath, ".json") {
		var jsonValue map[string]interface{}
		if err := json.Unmarshal(data, &jsonValue); err == nil {
			// Try to extract chat_template field
			if templateValue, ok := jsonValue["chat_template"]; ok {
				if templateStr, ok := templateValue.(string); ok {
					return &templateStr, nil
				}
			}
			// If root is a string, use it directly
			if templateStr, ok := jsonValue["template"].(string); ok {
				return &templateStr, nil
			}
		}
		// If JSON parsing fails, treat content as template string
	}

	// Clean up template (replace escaped newlines)
	content = strings.ReplaceAll(content, "\\n", "\n")

	return &content, nil
}

// discoverChatTemplate tries to discover chat template from tokenizer directory
func discoverChatTemplate(tokenizerPath string) *string {
	dir := filepath.Dir(tokenizerPath)

	// Try tokenizer_config.json first
	configPath := filepath.Join(dir, "tokenizer_config.json")
	if _, err := os.Stat(configPath); err == nil {
		template, err := LoadChatTemplateFromConfig(configPath)
		if err == nil && template != nil {
			return template
		}
	}

	// Try chat_template.json or chat_template.jinja
	candidates := []string{
		filepath.Join(dir, "chat_template.json"),
		filepath.Join(dir, "chat_template.jinja"),
	}

	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			template, err := loadChatTemplateFromFile(candidate)
			if err == nil && template != nil {
				return template
			}
		}
	}

	return nil
}
