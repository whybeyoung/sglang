package tokenizer

// SequenceDecoderOutput represents the output from processing a token
// Similar to Rust SequenceDecoderOutput
type SequenceDecoderOutput struct {
	Type  OutputType
	Text  string // Text content (for Text or StoppedWithText)
	Error error  // Error if processing failed
}

// OutputType represents the type of decoder output
type OutputType int

const (
	// OutputTypeText indicates text output (token decoded to text)
	OutputTypeText OutputType = iota
	// OutputTypeStoppedWithText indicates stopped with text (stop sequence matched, include text)
	OutputTypeStoppedWithText
	// OutputTypeStopped indicates stopped (stop sequence matched, no text)
	OutputTypeStopped
	// OutputTypeHeld indicates held (token held in buffer, not yet decoded)
	OutputTypeHeld
)

// StopSequenceDecoder processes tokens and handles stop sequences
// Similar to Rust StopSequenceDecoder
// NOTE: This is a simplified interface - full implementation requires
// sophisticated token-by-token processing and buffering
type StopSequenceDecoder interface {
	// ProcessToken processes a single token and returns output
	// Similar to Rust process_token method
	ProcessToken(tokenID uint32) (*SequenceDecoderOutput, error)

	// ProcessTokens processes multiple tokens and returns outputs
	// Similar to Rust process_tokens method
	ProcessTokens(tokenIDs []uint32) ([]*SequenceDecoderOutput, error)

	// Reset resets the decoder state (for reuse across multiple requests)
	Reset()

	// Flush returns any remaining text from the decoder buffer
	// Similar to Rust flush method
	Flush() *SequenceDecoderOutput
}

// StopSequenceDecoderBuilder builds a StopSequenceDecoder
// Similar to Rust StopSequenceDecoderBuilder
type StopSequenceDecoderBuilder struct {
	tokenizer         Tokenizer
	stopSequences     []string
	stopTokenIDs      []uint32
	skipSpecialTokens bool
	noStopTrim        bool
}

// NewStopSequenceDecoderBuilder creates a new builder
func NewStopSequenceDecoderBuilder(tokenizer Tokenizer) *StopSequenceDecoderBuilder {
	return &StopSequenceDecoderBuilder{
		tokenizer:         tokenizer,
		stopSequences:     []string{},
		stopTokenIDs:      []uint32{},
		skipSpecialTokens: true,
		noStopTrim:        false,
	}
}

// SkipSpecialTokens sets whether to skip special tokens
func (b *StopSequenceDecoderBuilder) SkipSpecialTokens(skip bool) *StopSequenceDecoderBuilder {
	b.skipSpecialTokens = skip
	return b
}

// StopSequence adds a stop sequence (hidden, will be trimmed)
func (b *StopSequenceDecoderBuilder) StopSequence(seq string) *StopSequenceDecoderBuilder {
	b.stopSequences = append(b.stopSequences, seq)
	return b
}

// VisibleStopSequence adds a visible stop sequence (will not be trimmed)
func (b *StopSequenceDecoderBuilder) VisibleStopSequence(seq string) *StopSequenceDecoderBuilder {
	// In visible mode, we track it but don't trim it
	// TODO: Implement visible stop sequence handling
	b.stopSequences = append(b.stopSequences, seq)
	return b
}

// StopToken adds a stop token ID (hidden, will be trimmed)
func (b *StopSequenceDecoderBuilder) StopToken(tokenID uint32) *StopSequenceDecoderBuilder {
	b.stopTokenIDs = append(b.stopTokenIDs, tokenID)
	return b
}

// VisibleStopToken adds a visible stop token ID (will not be trimmed)
func (b *StopSequenceDecoderBuilder) VisibleStopToken(tokenID uint32) *StopSequenceDecoderBuilder {
	// TODO: Implement visible stop token handling
	b.stopTokenIDs = append(b.stopTokenIDs, tokenID)
	return b
}

// Build creates a StopSequenceDecoder from the builder
func (b *StopSequenceDecoderBuilder) Build() StopSequenceDecoder {
	// Separate visible and hidden stops
	var stopSeqs, visibleStopSeqs []string
	var stopTokenIDs, visibleStopTokenIDs []uint32

	// For now, all stops are treated the same
	// In full implementation, we'd track which are visible vs hidden
	stopSeqs = b.stopSequences
	stopTokenIDs = b.stopTokenIDs

	return &SimpleStopDecoder{
		tokenizer:            b.tokenizer,
		stopSequences:        stopSeqs,
		stopTokenIDs:         stopTokenIDs,
		visibleStopSequences: visibleStopSeqs,
		visibleStopTokenIDs:  visibleStopTokenIDs,
		skipSpecialTokens:    b.skipSpecialTokens,
		noStopTrim:           b.noStopTrim,
		jailBuffer:           "",
		stopped:              false,
	}
}

// SimpleStopDecoder is an optimized implementation of StopSequenceDecoder
// Similar to Rust StopSequenceDecoder with improved sequence matching
type SimpleStopDecoder struct {
	tokenizer            Tokenizer
	stopSequences        []string
	stopTokenIDs         []uint32
	visibleStopSequences []string // Separate list for visible stops
	visibleStopTokenIDs  []uint32
	skipSpecialTokens    bool
	noStopTrim           bool
	jailBuffer           string // Buffer for accumulated text (similar to Rust jail_buffer)
	stopped              bool   // Whether we've stopped
}

// ProcessToken processes a single token with optimized sequence matching
// Similar to Rust StopSequenceDecoder::process_token
func (d *SimpleStopDecoder) ProcessToken(tokenID uint32) (*SequenceDecoderOutput, error) {
	// If already stopped, return Stopped
	if d.stopped {
		return &SequenceDecoderOutput{Type: OutputTypeStopped}, nil
	}

	// Check for stop token IDs first (before decoding)
	for _, stopID := range d.stopTokenIDs {
		if tokenID == stopID {
			d.stopped = true
			// Flush any jailed text before stopping
			if d.jailBuffer != "" {
				output := d.jailBuffer
				d.jailBuffer = ""
				return &SequenceDecoderOutput{
					Type: OutputTypeStoppedWithText,
					Text: output,
				}, nil
			}
			return &SequenceDecoderOutput{Type: OutputTypeStopped}, nil
		}
	}

	// Check for visible stop token IDs
	for _, visibleStopID := range d.visibleStopTokenIDs {
		if tokenID == visibleStopID {
			d.stopped = true
			// Include jailed text plus the stop token
			if d.tokenizer != nil {
				stopText, err := d.tokenizer.Decode([]uint32{tokenID}, d.skipSpecialTokens)
				if err != nil {
					stopText = ""
				}
				output := d.jailBuffer + stopText
				d.jailBuffer = ""
				return &SequenceDecoderOutput{
					Type: OutputTypeStoppedWithText,
					Text: output,
				}, nil
			}
			output := d.jailBuffer
			d.jailBuffer = ""
			return &SequenceDecoderOutput{
				Type: OutputTypeStoppedWithText,
				Text: output,
			}, nil
		}
	}

	// Decode token and append to sequence
	if d.tokenizer == nil {
		// No tokenizer - hold the token
		return &SequenceDecoderOutput{Type: OutputTypeHeld}, nil
	}

	// Decode the new token (incrementally)
	newText, err := d.tokenizer.Decode([]uint32{tokenID}, d.skipSpecialTokens)
	if err != nil {
		// If decode fails, hold the token
		return &SequenceDecoderOutput{Type: OutputTypeHeld}, nil
	}

	// Append to jail buffer
	d.jailBuffer += newText

	// Check for hidden stop sequences (find position of stop sequence in buffer)
	for _, stopSeq := range d.stopSequences {
		if pos := findString(d.jailBuffer, stopSeq); pos >= 0 {
			d.stopped = true
			// Return text before stop sequence
			output := d.jailBuffer[:pos]
			d.jailBuffer = ""
			if output == "" {
				return &SequenceDecoderOutput{Type: OutputTypeStopped}, nil
			}
			return &SequenceDecoderOutput{
				Type: OutputTypeStoppedWithText,
				Text: output,
			}, nil
		}
	}

	// Check for visible stop sequences
	for _, visibleSeq := range d.visibleStopSequences {
		if pos := findString(d.jailBuffer, visibleSeq); pos >= 0 {
			d.stopped = true
			// Return text including stop sequence
			endPos := pos + len(visibleSeq)
			output := d.jailBuffer[:endPos]
			d.jailBuffer = ""
			return &SequenceDecoderOutput{
				Type: OutputTypeStoppedWithText,
				Text: output,
			}, nil
		}
	}

	// Check for partial matches: is the end of jail_buffer the start of any stop_seq?
	// This handles stop sequences split across tokens
	bestSplitPos := d.findBestPartialMatch()
	if bestSplitPos >= 0 {
		// Hold the partial match, flush the rest
		toOutput := d.jailBuffer[:bestSplitPos]
		d.jailBuffer = d.jailBuffer[bestSplitPos:]
		if toOutput == "" {
			return &SequenceDecoderOutput{Type: OutputTypeHeld}, nil
		}
		return &SequenceDecoderOutput{
			Type: OutputTypeText,
			Text: toOutput,
		}, nil
	}

	// No partial matches - flush everything
	output := d.jailBuffer
	d.jailBuffer = ""
	if output == "" {
		return &SequenceDecoderOutput{Type: OutputTypeHeld}, nil
	}
	return &SequenceDecoderOutput{
		Type: OutputTypeText,
		Text: output,
	}, nil
}

// findBestPartialMatch finds the best split position for partial stop sequence matches
// Similar to Rust's partial match logic
func (d *SimpleStopDecoder) findBestPartialMatch() int {
	bufferLen := len(d.jailBuffer)
	if bufferLen == 0 {
		return -1
	}

	bestSplitPos := -1

	// Check all stop sequences (both hidden and visible)
	allStopSeqs := append([]string{}, d.stopSequences...)
	allStopSeqs = append(allStopSeqs, d.visibleStopSequences...)

	for _, stopSeq := range allStopSeqs {
		stopLen := len(stopSeq)
		if stopLen <= 1 {
			continue
		}

		maxLen := bufferLen
		if maxLen > stopLen-1 {
			maxLen = stopLen - 1
		}

		// Try decreasing lengths to find the longest matching prefix
		for suffixLen := maxLen; suffixLen >= 1; suffixLen-- {
			suffixStart := bufferLen - suffixLen
			// Check if this is a valid UTF-8 boundary
			if suffixStart >= len(d.jailBuffer) {
				continue
			}

			suffix := d.jailBuffer[suffixStart:]

			// Check if stop sequence starts with this suffix
			if len(stopSeq) >= suffixLen && stopSeq[:suffixLen] == suffix {
				// Found a partial match
				if bestSplitPos < 0 || suffixStart < bestSplitPos {
					bestSplitPos = suffixStart
				}
				break // Take the longest match for this sequence
			}
		}
	}

	return bestSplitPos
}

// findString finds substring position (simpler than strings.Contains for our use)
func findString(s, substr string) int {
	return findStringInRange(s, substr, 0, len(s))
}

// findStringInRange finds substring in a specific range
func findStringInRange(s, substr string, start, end int) int {
	if start < 0 || end > len(s) || start >= end {
		return -1
	}
	substrLen := len(substr)
	if substrLen == 0 {
		return start
	}
	if end-start < substrLen {
		return -1
	}

	// Simple substring search
	for i := start; i <= end-substrLen; i++ {
		if s[i:i+substrLen] == substr {
			return i
		}
	}
	return -1
}

// ProcessTokens processes multiple tokens
func (d *SimpleStopDecoder) ProcessTokens(tokenIDs []uint32) ([]*SequenceDecoderOutput, error) {
	outputs := make([]*SequenceDecoderOutput, 0, len(tokenIDs))
	for _, tokenID := range tokenIDs {
		output, err := d.ProcessToken(tokenID)
		if err != nil {
			return nil, err
		}
		outputs = append(outputs, output)
	}
	return outputs, nil
}

// Reset resets the decoder state
func (d *SimpleStopDecoder) Reset() {
	d.jailBuffer = ""
	d.stopped = false
}

// Flush returns any remaining text from the buffer
// Similar to Rust StopSequenceDecoder::flush
func (d *SimpleStopDecoder) Flush() *SequenceDecoderOutput {
	if d.jailBuffer != "" {
		output := d.jailBuffer
		d.jailBuffer = ""
		return &SequenceDecoderOutput{
			Type: OutputTypeText,
			Text: output,
		}
	}
	return &SequenceDecoderOutput{
		Type: OutputTypeText,
		Text: "",
	}
}

// CreateStopDecoder creates a StopSequenceDecoder from parameters
// Similar to Rust utils::create_stop_decoder
func CreateStopDecoder(
	tok Tokenizer,
	stopSequences []string,
	stopTokenIDs []uint32,
	skipSpecialTokens bool,
	noStopTrim bool,
) StopSequenceDecoder {
	builder := NewStopSequenceDecoderBuilder(tok).
		SkipSpecialTokens(skipSpecialTokens)

	for _, seq := range stopSequences {
		if noStopTrim {
			builder.VisibleStopSequence(seq)
		} else {
			builder.StopSequence(seq)
		}
	}

	for _, tokenID := range stopTokenIDs {
		if noStopTrim {
			builder.VisibleStopToken(tokenID)
		} else {
			builder.StopToken(tokenID)
		}
	}

	return builder.Build()
}

// ProcessChunkTokens processes a chunk of tokens through a stop decoder
// Similar to Rust StreamingProcessor::process_chunk_tokens
func ProcessChunkTokens(
	decoder StopSequenceDecoder,
	tokenIDs []uint32,
) (string, bool) {
	var chunkText string
	var shouldStop bool

	for _, tokenID := range tokenIDs {
		output, err := decoder.ProcessToken(tokenID)
		if err != nil {
			// On error, treat as Held (similar to Rust)
			continue
		}

		switch output.Type {
		case OutputTypeText:
			chunkText += output.Text
		case OutputTypeStoppedWithText:
			chunkText += output.Text
			shouldStop = true
			return chunkText, shouldStop
		case OutputTypeStopped:
			shouldStop = true
			return chunkText, shouldStop
		case OutputTypeHeld:
			// Continue processing
		}
	}

	return chunkText, shouldStop
}
