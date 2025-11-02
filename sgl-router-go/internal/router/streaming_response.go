package router

import (
	"io"
)

// StreamingResponse represents a streaming SSE response
// Similar to Rust's axum::Response for streaming
// The HTTP server should handle this by writing SSE events
type StreamingResponse struct {
	// Reader for SSE event stream
	// HTTP server should read from this and write to http.ResponseWriter
	Reader io.Reader

	// ContentType should be "text/event-stream"
	ContentType string

	// Additional headers
	Headers map[string]string
}

// NewStreamingResponse creates a new streaming response
func NewStreamingResponse(reader io.Reader) *StreamingResponse {
	return &StreamingResponse{
		Reader:      reader,
		ContentType: "text/event-stream",
		Headers: map[string]string{
			"Cache-Control": "no-cache",
			"Connection":    "keep-alive",
		},
	}
}
