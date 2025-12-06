// Package sglang provides a Go SDK for SGLang gRPC API.
//
// SGLang is a fast language model serving framework. This package provides a Go client
// library for interacting with SGLang's gRPC API, following the style of OpenAI's Go SDK.
//
// Basic usage:
//
//	client, err := sglang.NewClient(sglang.ClientConfig{
//		Endpoint:      "grpc://localhost:20000",
//		TokenizerPath: "/path/to/tokenizer",
//	})
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer client.Close()
//
//	resp, err := client.CreateChatCompletion(ctx, sglang.ChatCompletionRequest{
//		Model: "default",
//		Messages: []sglang.ChatMessage{
//			{Role: "user", Content: "Hello"},
//		},
//	})
//
// For streaming responses, use CreateChatCompletionStream instead.
package sglang

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/sglang/sglang-go-grpc-sdk/internal/ffi"
	grpcclient "github.com/sglang/sglang-go-grpc-sdk/internal/grpc"
)

// Client is the main client for interacting with SGLang gRPC API.
// It manages the connection to the SGLang server and handles both streaming
// and non-streaming chat completions.
//
// Thread-safe: All public methods are safe for concurrent use.
type Client struct {
	endpoint      string
	tokenizerPath string
	clientHandle  *ffi.SglangClientHandle // FFI-based client (legacy)
	grpcClient    *grpcclient.GrpcClient  // gRPC-based client (optimized)
	useGrpcClient bool                    // Whether to use gRPC client
	mu            sync.RWMutex
}

// ClientConfig holds configuration for creating a new client.
type ClientConfig struct {
	// Endpoint is the gRPC endpoint URL (e.g., "grpc://localhost:20000").
	// Required field. Must include the scheme (grpc://) and port number.
	Endpoint string

	// TokenizerPath is the path to the tokenizer directory containing
	// tokenizer configuration files (e.g., tokenizer.json, vocab.json).
	// Required field.
	TokenizerPath string

	// UseGrpcClient enables the optimized gRPC client mode.
	// When true, uses direct gRPC calls with batch postprocessing (reduces FFI overhead by 90%+).
	// When false (default), uses the traditional FFI-based client for backward compatibility.
	// Default: false
	UseGrpcClient bool
}

// NewClient creates a new SGLang client with the given configuration.
//
// The client maintains a long-lived connection to the SGLang server and should
// be reused for multiple requests. Call Close() to release resources.
//
// Returns an error if:
// - Endpoint is empty
// - TokenizerPath is empty
// - Connection to the server fails
func NewClient(config ClientConfig) (*Client, error) {
	if config.Endpoint == "" {
		return nil, errors.New("endpoint is required")
	}
	if config.TokenizerPath == "" {
		return nil, errors.New("tokenizer path is required")
	}

	client := &Client{
		endpoint:      config.Endpoint,
		tokenizerPath: config.TokenizerPath,
		useGrpcClient: config.UseGrpcClient,
	}

	if config.UseGrpcClient {
		// Use optimized gRPC client
		grpcClient, err := grpcclient.NewGrpcClient(config.Endpoint, config.TokenizerPath)
		if err != nil {
			return nil, fmt.Errorf("failed to create gRPC client: %w", err)
		}
		client.grpcClient = grpcClient
	} else {
		// Use legacy FFI client (backward compatible)
		clientHandle, err := ffi.NewClient(config.Endpoint, config.TokenizerPath)
		if err != nil {
			return nil, fmt.Errorf("failed to create client: %w", err)
		}
		client.clientHandle = clientHandle
	}

	return client, nil
}

// Close closes the client and releases all resources.
//
// After Close() is called, the client cannot be used for further requests.
// Calling Close() multiple times is safe and idempotent.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.grpcClient != nil {
		if err := c.grpcClient.Close(); err != nil {
			return err
		}
		c.grpcClient = nil
	}

	if c.clientHandle != nil {
		c.clientHandle.Free()
		c.clientHandle = nil
	}
	return nil
}

// ChatCompletionRequest represents a request for chat completion.
// It follows the OpenAI API style for familiar usage.
type ChatCompletionRequest struct {
	// Model specifies the model to use for completion (e.g., "default")
	Model string `json:"model"`
	// Messages is the list of messages in the conversation
	Messages            []ChatMessage   `json:"messages"`
	Temperature         *float32        `json:"temperature,omitempty"`
	TopP                *float32        `json:"top_p,omitempty"`
	TopK                *int            `json:"top_k,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	Stream              bool            `json:"stream"`
	Tools               []Tool          `json:"tools,omitempty"`
	ToolChoice          interface{}     `json:"tool_choice,omitempty"`
	Stop                interface{}     `json:"stop,omitempty"`
	StopTokenIDs        []int           `json:"stop_token_ids,omitempty"`
	SkipSpecialTokens   bool            `json:"skip_special_tokens,omitempty"`
	FrequencyPenalty    *float32        `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float32        `json:"presence_penalty,omitempty"`
	ResponseFormat      *ResponseFormat `json:"response_format,omitempty"`
	Seed                *int            `json:"seed,omitempty"`
	Logprobs            bool            `json:"logprobs,omitempty"`
	TopLogprobs         *int            `json:"top_logprobs,omitempty"`
	User                string          `json:"user,omitempty"`
}

// ChatMessage represents a single message in a chat conversation
type ChatMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
	Name    string      `json:"name,omitempty"`
}

// Tool represents a tool/function that can be called
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function represents a function definition
type Function struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// ResponseFormat represents the response format
type ResponseFormat struct {
	Type string `json:"type"`
}

// ChatCompletionResponse represents a non-streaming chat completion response
type ChatCompletionResponse struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
	Choices           []Choice `json:"choices"`
	Usage             Usage    `json:"usage"`
}

// Choice represents a choice in the completion response
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Message represents a message in the response
type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ToolCall represents a tool call in the response
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall represents a function call
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Usage represents token usage information
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionStreamResponse represents a streaming chat completion response
type ChatCompletionStreamResponse struct {
	ID                string         `json:"id"`
	Object            string         `json:"object"`
	Created           int64          `json:"created"`
	Model             string         `json:"model"`
	SystemFingerprint string         `json:"system_fingerprint,omitempty"`
	Choices           []StreamChoice `json:"choices"`
	Usage             *Usage         `json:"usage,omitempty"`
}

// StreamChoice represents a choice in a streaming response
type StreamChoice struct {
	Index        int          `json:"index"`
	Delta        MessageDelta `json:"delta"`
	FinishReason string       `json:"finish_reason,omitempty"`
}

// MessageDelta represents incremental message updates
type MessageDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// CreateChatCompletion creates a non-streaming chat completion with context support.
//
// Context Support:
// The ctx parameter is fully supported for cancellation and timeouts:
// - If ctx is cancelled, the request will be interrupted on the next stream.Recv() call
// - If ctx times out, the request will return context.DeadlineExceeded
//
// Example with timeout:
//
//	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
//	defer cancel()
//	resp, err := client.CreateChatCompletion(ctx, req)
//
// Note: Internally, this creates a stream and collects all chunks,
// so context monitoring happens at the chunk level.
func (c *Client) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	// For non-streaming, we'll collect all chunks and return the final response
	req.Stream = true // We still use streaming internally, but collect all chunks

	// Prepare request: if Tools is empty, set to nil for proper JSON serialization
	if len(req.Tools) == 0 {
		req.Tools = nil
	}

	stream, err := c.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, err
	}
	defer stream.Close()

	var fullContent strings.Builder
	var fullToolCalls []ToolCall
	var finishReason string
	var usage Usage
	var responseID string
	var created int64
	var model string
	var systemFingerprint string

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		if chunk.ID != "" {
			responseID = chunk.ID
		}
		if chunk.Created > 0 {
			created = chunk.Created
		}
		if chunk.Model != "" {
			model = chunk.Model
		}
		if chunk.SystemFingerprint != "" {
			systemFingerprint = chunk.SystemFingerprint
		}

		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				fullContent.WriteString(choice.Delta.Content)
			}
			if len(choice.Delta.ToolCalls) > 0 {
				fullToolCalls = append(fullToolCalls, choice.Delta.ToolCalls...)
			}
			// Always update finish_reason if present (even if empty string, but should not be empty)
			// The last chunk (Complete message) should have finish_reason set
			if choice.FinishReason != "" {
				finishReason = choice.FinishReason
			}
		}

		// Extract usage from chunk if available (usually in the last chunk)
		// Always update usage if present, as the last chunk should have the final usage
		if chunk.Usage != nil {
			usage = *chunk.Usage
		}
	}

	// Build final response
	message := Message{
		Role:    "assistant",
		Content: fullContent.String(),
	}
	if len(fullToolCalls) > 0 {
		message.ToolCalls = fullToolCalls
	}

	// Ensure finish_reason is set (defensive check)
	// If finish_reason is still empty, default to "stop"
	if finishReason == "" {
		finishReason = "stop"
	}

	return &ChatCompletionResponse{
		ID:                responseID,
		Object:            "chat.completion",
		Created:           created,
		Model:             model,
		SystemFingerprint: systemFingerprint,
		Choices: []Choice{
			{
				Index:        0,
				Message:      message,
				FinishReason: finishReason,
			},
		},
		Usage: usage,
	}, nil
}

// ChatCompletionStream represents a streaming chat completion
type ChatCompletionStream struct {
	stream     *ffi.SglangStreamHandle              // FFI stream (legacy mode)
	grpcStream *grpcclient.GrpcChatCompletionStream // gRPC stream (optimized mode)
	// Removed mu sync.Mutex - using atomic operations and channel-based error passing
	done   int32              // Track if stream has been marked as done (using atomic, like GrpcChatCompletionStream)
	ctx    context.Context    // Context for cancellation support
	cancel context.CancelFunc // Cancel function to stop monitoring goroutine
	closed chan struct{}      // Signal when stream is closed

	// Async optimization: background goroutine reads from FFI/gRPC and sends to channel
	chunks chan chunkResult // Channel for receiving chunks from background goroutine
	// Removed err error - errors are passed through chunks channel, no need to store separately
}

// chunkResult represents a chunk read from the stream or an error
type chunkResult struct {
	chunk *ChatCompletionStreamResponse
	err   error
}

// Recv receives the next chunk from the stream.
//
// Supports context cancellation: if the context passed to CreateChatCompletionStream
// is cancelled, Recv will return context.Canceled error on the next call.
//
// Optimization: Uses a background goroutine to read from FFI, avoiding block_on
// overhead on each Recv() call. Recv() simply reads from a buffered channel.
func (s *ChatCompletionStream) Recv() (*ChatCompletionStreamResponse, error) {
	// Check if context was cancelled
	select {
	case <-s.ctx.Done():
		return nil, s.ctx.Err() // Returns context.Canceled or context.DeadlineExceeded
	default:
	}

	// Read from channel (non-blocking check first)
	select {
	case <-s.ctx.Done():
		return nil, s.ctx.Err()
	case result, ok := <-s.chunks:
		if !ok {
			// Channel closed - errors are already passed through chunks channel
			// No need to check stored error (removed err field)
			return nil, io.EOF
		}
		if result.err != nil {
			return nil, result.err
		}
		return result.chunk, nil
	}
}

// RecvJSON receives the next chunk as raw JSON string (optimized path to avoid parsing/serialization)
// This is much faster than Recv() because it avoids JSON unmarshal/marshal overhead
// Use this when you only need to forward the JSON to the client (e.g., SSE streaming)
func (s *ChatCompletionStream) RecvJSON() (string, error) {
	// If using gRPC stream, use its optimized RecvJSON method
	if s.grpcStream != nil {
		return s.grpcStream.RecvJSON()
	}

	// For FFI stream, we need to parse and re-serialize (no direct JSON path)
	// This is less optimal, but maintains compatibility
	chunk, err := s.Recv()
	if err != nil {
		return "", err
	}
	jsonData, err := json.Marshal(chunk)
	if err != nil {
		return "", fmt.Errorf("failed to marshal response: %w", err)
	}
	return string(jsonData), nil
}

// Close closes the stream and cancels any pending operations.
func (s *ChatCompletionStream) Close() error {
	// Mark as done to stop readLoop (using atomic, like GrpcChatCompletionStream)
	atomic.StoreInt32(&s.done, 1)

	// Read stream handles (Close() is rarely called, so no lock needed)
	// These are only set to nil in Close(), and Close() should only be called once
	streamHandle := s.stream
	grpcStreamHandle := s.grpcStream

	// Cancel the context to signal the readLoop goroutine to stop
	if s.cancel != nil {
		s.cancel()
	}

	// Signal that stream is closed
	select {
	case <-s.closed:
		// Already closed
	default:
		close(s.closed)
	}

	// Close gRPC stream if using gRPC mode
	if grpcStreamHandle != nil {
		if err := grpcStreamHandle.Close(); err != nil {
			return err
		}
		// Set to nil (Close() is rarely called, so no lock needed)
		s.grpcStream = nil
	}

	// Free the FFI stream if using FFI mode
	// This prevents AbortOnDropStream from sending abort when dropped
	if streamHandle != nil {
		streamHandle.Free()
		// Set to nil (Close() is rarely called, so no lock needed)
		s.stream = nil
	}

	// Wait a bit for readLoop to exit (it will close the channel)
	// The channel will be closed by readLoop's defer, so we don't need to drain it
	// The readLoop will exit when it sees s.closed or s.ctx.Done()
	return nil
}

// CreateChatCompletionStream creates a streaming chat completion with context cancellation support.
//
// Context Support:
// The ctx parameter is now fully supported for cancellation and timeouts:
// - If ctx is cancelled, stream.Recv() will return context.Canceled on the next call
// - If ctx times out (WithTimeout), stream.Recv() will return context.DeadlineExceeded
// - Calling stream.Close() also cancels the context
//
// Example with timeout:
//
//	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
//	defer cancel()
//	stream, err := client.CreateChatCompletionStream(ctx, req)
//	// Stream will auto-close if 30 seconds elapse
//
// Example with cancellation:
//
//	ctx, cancel := context.WithCancel(context.Background())
//	stream, err := client.CreateChatCompletionStream(ctx, req)
//	go func() {
//	    time.Sleep(5*time.Second)
//	    cancel()  // Cancel after 5 seconds
//	}()
func (c *Client) CreateChatCompletionStream(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionStream, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// Marshal request to JSON, then ensure tools field is always present.
	// Due to omitempty tag, empty Tools slice will be omitted from JSON.
	// We need to ensure tools field is always present as [] when empty (not omitted),
	// matching the behavior of complete_sdk example.
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Unmarshal into map and ensure tools field is present
	var reqMap map[string]interface{}
	if err := json.Unmarshal(reqJSON, &reqMap); err != nil {
		return nil, fmt.Errorf("failed to unmarshal request to map: %w", err)
	}

	// Add empty tools array if not present
	if _, exists := reqMap["tools"]; !exists {
		reqMap["tools"] = []interface{}{}
	}

	// Marshal back to JSON
	reqJSON, err = json.Marshal(reqMap)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request map to JSON: %w", err)
	}

	// Choose client implementation based on configuration
	if c.useGrpcClient {
		if c.grpcClient == nil {
			return nil, errors.New("gRPC client is closed")
		}

		// Use optimized gRPC client
		grpcStream, err := c.grpcClient.CreateChatCompletionStream(ctx, string(reqJSON))
		if err != nil {
			return nil, fmt.Errorf("failed to create gRPC stream: %w", err)
		}

		// Wrap gRPC stream to match ChatCompletionStream interface
		return wrapGrpcStream(grpcStream, ctx), nil
	} else {
		// Use legacy FFI client (backward compatible)
		if c.clientHandle == nil {
			return nil, errors.New("client is closed")
		}

		// Create stream
		streamHandle, err := c.clientHandle.ChatCompletionStream(string(reqJSON))
		if err != nil {
			return nil, fmt.Errorf("failed to create stream: %w", err)
		}

		// Create a child context from the provided context for cancellation support
		streamCtx, cancel := context.WithCancel(ctx)

		// Create buffered channel for async chunk delivery
		// Buffer size: 200 chunks to reduce blocking in high concurrency scenarios
		// FFI mode uses block_on which can block, so larger buffer helps prevent deadlocks
		chunksChan := make(chan chunkResult, 200)

		stream := &ChatCompletionStream{
			stream: streamHandle,
			ctx:    streamCtx,
			cancel: cancel,
			closed: make(chan struct{}),
			chunks: chunksChan,
		}

		// Start background goroutine to read from FFI stream
		// This avoids block_on overhead on each Recv() call
		go stream.readLoop()

		return stream, nil
	}
}

// wrapGrpcStream wraps a gRPC stream to match ChatCompletionStream interface
func wrapGrpcStream(grpcStream *grpcclient.GrpcChatCompletionStream, ctx context.Context) *ChatCompletionStream {
	streamCtx, cancel := context.WithCancel(ctx)
	chunksChan := make(chan chunkResult, 200) // Larger buffer for 20 concurrent requests (was 100)

	stream := &ChatCompletionStream{
		stream:     nil, // Not used for gRPC streams
		ctx:        streamCtx,
		cancel:     cancel,
		closed:     make(chan struct{}),
		chunks:     chunksChan,
		grpcStream: grpcStream, // Store gRPC stream reference
	}

	// Start background goroutine to read from gRPC stream
	go stream.readGrpcLoop(grpcStream)

	return stream
}

// readGrpcLoop reads from gRPC stream and sends chunks to channel
func (s *ChatCompletionStream) readGrpcLoop(grpcStream *grpcclient.GrpcChatCompletionStream) {
	defer close(s.chunks)

	for {
		// Check if context was cancelled or stream was closed
		select {
		case <-s.ctx.Done():
			// Send error through channel (no need to store separately)
			select {
			case s.chunks <- chunkResult{err: s.ctx.Err()}:
			case <-s.closed:
			}
			return
		case <-s.closed:
			return
		default:
		}

		// Read from gRPC stream
		// This calls GrpcChatCompletionStream.Recv() which reads from resultChan
		grpcResp, err := grpcStream.Recv()
		if err != nil {
			if err == io.EOF {
				return
			}
			// Send error through channel (no need to store separately)
			select {
			case s.chunks <- chunkResult{err: err}:
			case <-s.ctx.Done():
				return
			case <-s.closed:
				return
			}
			return
		}

		// Check for nil response (shouldn't happen, but be defensive)
		if grpcResp == nil {
			continue
		}

		// Convert grpc.ChatCompletionStreamResponse to client.ChatCompletionStreamResponse
		// This preserves all fields including Usage
		response := convertGrpcResponse(grpcResp)

		// Send chunk to channel
		select {
		case s.chunks <- chunkResult{chunk: &response}:
			// Successfully sent to chunks channel
		case <-s.ctx.Done():
			return
		case <-s.closed:
			return
		}
	}
}

// convertGrpcResponse converts grpc.ChatCompletionStreamResponse to client.ChatCompletionStreamResponse
func convertGrpcResponse(grpcResp *grpcclient.ChatCompletionStreamResponse) ChatCompletionStreamResponse {
	// Handle nil pointer case
	if grpcResp == nil {
		return ChatCompletionStreamResponse{
			ID:      "",
			Object:  "",
			Created: 0,
			Model:   "",
			Choices: []StreamChoice{},
			Usage:   nil,
		}
	}

	// Determine choices length safely
	choicesLen := 0
	if grpcResp.Choices != nil {
		choicesLen = len(grpcResp.Choices)
	}

	response := ChatCompletionStreamResponse{
		ID:                grpcResp.ID,
		Object:            grpcResp.Object,
		Created:           grpcResp.Created,
		Model:             grpcResp.Model,
		SystemFingerprint: grpcResp.SystemFingerprint,
		Choices:           make([]StreamChoice, choicesLen),
		Usage:             nil,
	}

	// Convert choices (handle nil safely)
	if grpcResp.Choices != nil {
		for i, grpcChoice := range grpcResp.Choices {
			delta := MessageDelta{
				Role:    grpcChoice.Delta.Role,
				Content: grpcChoice.Delta.Content,
			}
			// Handle tool calls safely
			if grpcChoice.Delta.ToolCalls != nil {
				delta.ToolCalls = make([]ToolCall, len(grpcChoice.Delta.ToolCalls))
				for j, grpcToolCall := range grpcChoice.Delta.ToolCalls {
					delta.ToolCalls[j] = ToolCall{
						ID:   grpcToolCall.ID,
						Type: grpcToolCall.Type,
						Function: FunctionCall{
							Name:      grpcToolCall.Function.Name,
							Arguments: grpcToolCall.Function.Arguments,
						},
					}
				}
			}
			response.Choices[i] = StreamChoice{
				Index:        grpcChoice.Index,
				FinishReason: grpcChoice.FinishReason,
				Delta:        delta,
			}
		}
	}

	// Convert usage if present
	if grpcResp.Usage != nil {
		response.Usage = &Usage{
			PromptTokens:     grpcResp.Usage.PromptTokens,
			CompletionTokens: grpcResp.Usage.CompletionTokens,
			TotalTokens:      grpcResp.Usage.TotalTokens,
		}
	}

	return response
}

// readLoop runs in a background goroutine and continuously reads from the FFI stream.
// It sends chunks to the channel, avoiding block_on overhead in Recv().
//
// Optimization: This goroutine handles all FFI calls (which involve block_on),
// while Recv() simply reads from a buffered channel. This eliminates the block_on
// overhead from the critical path of Recv(), reducing latency and CPU usage.
func (s *ChatCompletionStream) readLoop() {
	defer close(s.chunks)

	for {
		// Check if context was cancelled or stream was closed
		select {
		case <-s.ctx.Done():
			// Send error through channel (no need to store separately)
			select {
			case s.chunks <- chunkResult{err: s.ctx.Err()}:
			case <-s.closed:
			}
			return
		case <-s.closed:
			return
		default:
		}

		// Check if stream handle is valid
		// Read handles and done flag (readLoop is single-threaded, so no lock needed)
		streamHandle := s.stream
		grpcStreamHandle := s.grpcStream
		done := atomic.LoadInt32(&s.done) == 1

		// If using gRPC mode, readGrpcLoop handles it separately
		if grpcStreamHandle != nil {
			return
		}

		if streamHandle == nil || done {
			return
		}

		// Read next chunk from FFI (this is where block_on happens, but only in this goroutine)
		// The block_on overhead is now isolated to this background goroutine, not in Recv()
		// WARNING: ReadNext() uses RUNTIME.block_on which will block waiting for data.
		// In high concurrency scenarios, multiple readLoop goroutines may block here.
		// Consider using gRPC mode (UseGrpcClient=true) for better performance and non-blocking behavior.
		responseJSON, isDone, err := streamHandle.ReadNext()
		if err != nil {
			// Send error through channel (no need to store separately)
			select {
			case s.chunks <- chunkResult{err: err}:
			case <-s.ctx.Done():
				return
			case <-s.closed:
				return
			}
			return
		}

		// Mark stream as done if ReadNext indicates completion (using atomic)
		if isDone {
			atomic.StoreInt32(&s.done, 1)
		}

		// If we have a response, parse and send it
		if responseJSON != "" {
			var response ChatCompletionStreamResponse
			if err := json.Unmarshal([]byte(responseJSON), &response); err != nil {
				// Send error through channel (no need to store separately)
				parseErr := fmt.Errorf("failed to parse response: %w", err)
				select {
				case s.chunks <- chunkResult{err: parseErr}:
				case <-s.ctx.Done():
					return
				case <-s.closed:
					return
				}
				return
			}

			// Send chunk to channel (non-blocking with timeout check)
			select {
			case s.chunks <- chunkResult{chunk: &response}:
				// Successfully sent chunk
			case <-s.ctx.Done():
				return
			case <-s.closed:
				return
			}
		}

		// If stream is done but no response, exit (channel will be closed by defer)
		if isDone {
			return
		}

		// Empty response and stream not done - continue loop to read next chunk
		// This handles Ok(None) cases where Rust returns no data but stream continues
	}
}
