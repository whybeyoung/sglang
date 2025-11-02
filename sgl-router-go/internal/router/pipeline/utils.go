package pipeline

import (
	"fmt"
)

// Helper utilities for pipeline processing
// Similar to Rust routers::grpc::utils

// BadRequestError creates a bad request error response
func BadRequestError(message string) error {
	return fmt.Errorf("bad request: %s", message)
}

// InternalError creates an internal error response
func InternalError(message string) error {
	return fmt.Errorf("internal error: %s", message)
}

// ServiceUnavailableError creates a service unavailable error response
func ServiceUnavailableError(message string) error {
	return fmt.Errorf("service unavailable: %s", message)
}

// ExtractModelID extracts model ID from request context
func ExtractModelID(ctx *RequestContext) string {
	if ctx.Input.ModelID != nil {
		return *ctx.Input.ModelID
	}

	// Try to extract from request based on type
	switch ctx.Input.RequestType {
	case RequestTypeChat:
		// TODO: Extract from ChatCompletionRequest
		return "default"
	case RequestTypeGenerate:
		// Generate requests don't have model field
		return "default"
	default:
		return "default"
	}
}

// IsStreamingRequest checks if the request is for streaming
func IsStreamingRequest(ctx *RequestContext) bool {
	if ctx.State.Dispatch != nil {
		return ctx.State.Dispatch.IsStreaming
	}
	// TODO: Extract from actual request
	return false
}
