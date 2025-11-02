package grpc

import (
	"context"
)

// GenerateRequest is the interface for proto GenerateRequest
// TODO: Replace with actual proto.GenerateRequest after make generate
type GenerateRequest interface {
	// GetRequestID returns the request ID
	GetRequestID() string
	// Clone creates a copy of the request
	Clone() GenerateRequest
}

// GenerateResponseChunk represents a chunk from the streaming response
// TODO: Replace with actual proto.GenerateResponse after make generate
type GenerateResponseChunk interface {
	// GetChunk returns the chunk data
	GetChunk() *ResponseChunk
	// GetComplete returns the complete response (if stream finished)
	GetComplete() *ResponseComplete
	// IsError returns true if this is an error response
	IsError() bool
	// GetErrorMessage returns error message if IsError() is true
	GetErrorMessage() string
}

// ResponseChunk represents a streaming chunk
type ResponseChunk struct {
	Index          uint32
	TokenIDs       []uint32
	Text           string      // Decoded text (if available)
	OutputLogprobs interface{} // TODO: Proper type after proto generation
}

// ResponseComplete represents a complete response
type ResponseComplete struct {
	Index            uint32
	OutputIDs        []uint32
	FinishReason     string
	MatchedStop      interface{} // TODO: Proper type after proto generation
	InputLogprobs    interface{} // TODO: Proper type after proto generation
	OutputLogprobs   interface{} // TODO: Proper type after proto generation
	PromptTokens     uint32
	CompletionTokens uint32
	CachedTokens     uint32
}

// GenerateStream is the interface for the gRPC Generate stream
// TODO: Replace with actual proto stream type after make generate
type GenerateStream interface {
	// Recv receives the next chunk from the stream
	Recv() (GenerateResponseChunk, error)
	// Context returns the stream context
	Context() context.Context
	// CloseSend closes the send direction of the stream
	CloseSend() error
}

// SglangSchedulerClient is the interface for the gRPC client
// TODO: Replace with actual proto client after make generate
type SglangSchedulerClient interface {
	// Generate starts a Generate RPC and returns a stream
	Generate(ctx context.Context, req GenerateRequest) (GenerateStream, error)
	// HealthCheck performs a health check
	HealthCheck(ctx context.Context) error
	// Close closes the client connection
	Close() error
}

// ProtoConverter converts between our protocol types and proto types
// This will be implemented after proto code generation
type ProtoConverter interface {
	// ToGenerateRequest converts a GenerateRequest to proto
	ToGenerateRequest(req *ProtocolsGenerateRequest) GenerateRequest
	// FromGenerateResponseChunk converts a proto chunk to our format
	FromGenerateResponseChunk(chunk GenerateResponseChunk) (*ResponseChunk, error)
}

// ProtocolsGenerateRequest is a placeholder for our protocol-level GenerateRequest
// This will be replaced with actual conversion logic
type ProtocolsGenerateRequest struct {
	// This is a placeholder - actual fields will be defined based on proto
}
