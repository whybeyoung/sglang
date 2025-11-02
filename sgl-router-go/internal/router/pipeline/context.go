package pipeline

import (
	"context"
	"io"
	"sync"

	"github.com/sglang/sglang-router-go/internal/core"
	"go.uber.org/zap"
)

// RequestType represents the type of request
type RequestType int

const (
	RequestTypeChat RequestType = iota
	RequestTypeGenerate
	RequestTypeResponses
)

// RequestContext is the main context that flows through the pipeline
// Similar to Rust RequestContext, this is the single source of truth for request state
type RequestContext struct {
	// Immutable input
	Input *RequestInput

	// Shared components (immutable references)
	Components *SharedComponents

	// Mutable processing state (evolves through pipeline)
	State *ProcessingState

	// Context for cancellation/timeout
	ctx context.Context

	// Logger for this request
	logger *zap.Logger
}

// RequestInput contains immutable request input
type RequestInput struct {
	RequestType RequestType
	ModelID     *string
	// Store the actual request object for parameter extraction
	// For Chat: *protocols.ChatCompletionRequest
	// For Generate: *protocols.GenerateRequest
	Request interface{} // Store actual request for parameter extraction
	// Note: In Rust, headers are stored here. In Go, we might use context instead
}

// SharedComponents contains shared components injected once at creation
type SharedComponents struct {
	// Tokenizer interface (simplified - see comments in tokenizer.go)
	Tokenizer interface{} // TODO: Define proper Tokenizer interface
	// Tool parser factory (simplified)
	ToolParserFactory interface{} // TODO: Define proper factory
	// Reasoning parser factory (simplified)
	ReasoningParserFactory interface{} // TODO: Define proper factory
}

// ProcessingState contains mutable state that evolves through pipeline stages
type ProcessingState struct {
	mu sync.RWMutex

	// Stage 1: Preparation outputs
	Preparation *PreparationOutput

	// Stage 2: Worker selection outputs
	Workers *WorkerSelection

	// Stage 3: Client acquisition outputs
	Clients *ClientSelection

	// Stage 4: Request building outputs
	ProtoRequest interface{} // *proto.GenerateRequest from pkg/proto

	// Stage 5: Dispatch metadata
	Dispatch *DispatchMetadata

	// Stage 6: Response processing state
	Response *ResponseState
}

// StreamingResponse represents a streaming SSE response
// Moved here to avoid import cycles between pipeline/stages and router packages
type StreamingResponse struct {
	Reader      io.Reader
	ContentType string
	Headers     map[string]string
}

// PreparationOutput contains outputs from the preparation stage
type PreparationOutput struct {
	OriginalText      *string
	TokenIDs          []uint32
	ProcessedMessages interface{} // TODO: Define proper type
	ToolConstraints   *ToolConstraints
	FilteredRequest   interface{} // For chat requests with filtered tools
}

// ToolConstraints represents tool call constraints
type ToolConstraints struct {
	Type  string
	Value string
}

// WorkerSelection represents selected worker(s)
type WorkerSelection struct {
	IsDual bool
	Single Worker
	Dual   struct {
		Prefill Worker
		Decode  Worker
	}
}

// Worker is an alias for core.Worker to avoid import cycles
type Worker = core.Worker

// ClientSelection represents selected gRPC client(s)
type ClientSelection struct {
	IsDual bool
	Single interface{} // TODO: Use proper gRPC client type
	Dual   struct {
		Prefill interface{}
		Decode  interface{}
	}
}

// DispatchMetadata contains dispatch metadata
type DispatchMetadata struct {
	RequestID     string
	Model         string
	Created       int64 // Unix timestamp
	WeightVersion *string
	IsStreaming   bool
}

// ResponseState contains response processing state
type ResponseState struct {
	StopDecoder       interface{} // TODO: Define proper stop decoder
	Streaming         *StreamingState
	Collected         interface{} // For non-streaming responses
	ExecutionResult   *ExecutionResult
	FinalResponse     interface{} // Final processed response (non-streaming)
	StreamingResponse interface{} // Streaming response (SSE, early pipeline exit)
}

// StreamingState tracks streaming state per choice/index
type StreamingState struct {
	mu sync.RWMutex
	// Maps index -> state
	IsFirsts         map[uint32]bool
	StreamBuffers    map[uint32]string
	FinishReasons    map[uint32]string
	MatchedStops     map[uint32]interface{}
	PromptTokens     map[uint32]uint32
	CompletionTokens map[uint32]uint32
	CachedTokens     map[uint32]uint32
}

// ExecutionResult represents the result of request execution
type ExecutionResult struct {
	IsDual bool
	Single interface{} // TODO: Use proper stream type
	Dual   struct {
		Prefill interface{}
		Decode  interface{}
	}
}

// NewRequestContext creates a new request context
func NewRequestContext(ctx context.Context, input *RequestInput, components *SharedComponents, logger *zap.Logger) *RequestContext {
	return &RequestContext{
		Input:      input,
		Components: components,
		State: &ProcessingState{
			Response: &ResponseState{
				Streaming: &StreamingState{
					IsFirsts:         make(map[uint32]bool),
					StreamBuffers:    make(map[uint32]string),
					FinishReasons:    make(map[uint32]string),
					MatchedStops:     make(map[uint32]interface{}),
					PromptTokens:     make(map[uint32]uint32),
					CompletionTokens: make(map[uint32]uint32),
					CachedTokens:     make(map[uint32]uint32),
				},
			},
		},
		ctx:    ctx,
		logger: logger,
	}
}

// Context returns the context for cancellation/timeout
func (rc *RequestContext) Context() context.Context {
	return rc.ctx
}
