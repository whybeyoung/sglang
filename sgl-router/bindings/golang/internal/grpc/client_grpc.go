// Package grpc provides gRPC client implementation for SGLang
package grpc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/sglang/sglang-go-grpc-sdk/internal/ffi"
	"github.com/sglang/sglang-go-grpc-sdk/internal/proto"
)

// grpcClientStream is an interface for gRPC client streams
// Note: The generated code uses grpc.ServerStreamingClient[GenerateResponse]
// which is a generic type, so we use a type assertion approach
type grpcClientStream interface {
	Recv() (*proto.GenerateResponse, error)
	CloseSend() error
}

// recvResult holds the result of a Recv() call
type recvResult struct {
	resp *proto.GenerateResponse
	err  error
}

// GrpcClient is a gRPC-based client for SGLang
// This client uses direct gRPC calls instead of FFI, reducing overhead
type GrpcClient struct {
	conn            *grpc.ClientConn
	client          proto.SglangSchedulerClient
	tokenizerPath   string
	tokenizerHandle *ffi.TokenizerHandle // Pre-created at startup, thread-safe for concurrent use
	// Removed mu: TokenizerHandle is pre-created at startup and thread-safe (Arc<dyn TokenizerTrait> with Send + Sync)
	// All tokenizer methods are read-only (&self), so concurrent calls are safe
	// No lock needed since tokenizer is created once at startup and never modified
}

// NewGrpcClient creates a new gRPC client and initializes the tokenizer
// The tokenizer is created at startup time to avoid first-request latency
func NewGrpcClient(endpoint, tokenizerPath string) (*GrpcClient, error) {
	// Parse endpoint (format: grpc://host:port)
	endpoint = strings.TrimPrefix(endpoint, "grpc://")
	if !strings.Contains(endpoint, ":") {
		return nil, fmt.Errorf("invalid endpoint format: %s (expected grpc://host:port)", endpoint)
	}

	// Create gRPC connection with keepalive settings to avoid "Too many pings" error
	// Keepalive settings:
	// - Time: Send ping every 120 seconds if there's no activity (very conservative to avoid "too_many_pings")
	// - Timeout: Wait 20 seconds for ping ack before considering connection dead
	// - PermitWithoutStream: false - Only ping when there are active streams (reduces ping frequency)
	// Note: Some servers have very strict ping limits, so we use very conservative settings
	// If "too_many_pings" error still persists, we may need to disable keepalive entirely
	keepaliveParams := keepalive.ClientParameters{
		Time:                120 * time.Second, // Send ping if no activity for 120s (very conservative)
		Timeout:             20 * time.Second,  // Wait 20s for ping ack
		PermitWithoutStream: false,             // Only ping when there are active streams (reduces ping frequency)
	}

	// Build client options
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepaliveParams),
	}

	conn, err := grpc.NewClient(endpoint, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to gRPC server: %w", err)
	}

	client := proto.NewSglangSchedulerClient(conn)

	// Create tokenizer handle at startup time to avoid first-request latency
	// This eliminates the need for double-check locking in request handlers
	tokenizerHandle, err := ffi.CreateTokenizerHandle(tokenizerPath)
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to create tokenizer handle: %w", err)
	}

	return &GrpcClient{
		conn:            conn,
		client:          client,
		tokenizerPath:   tokenizerPath,
		tokenizerHandle: tokenizerHandle, // Pre-created at startup
	}, nil
}

// Close closes the gRPC connection and frees the pre-created tokenizer
func (c *GrpcClient) Close() error {
	// Free pre-created tokenizer
	// No lock needed - Close() should only be called when no concurrent requests are in flight
	if c.tokenizerHandle != nil {
		ffi.FreeTokenizerHandle(c.tokenizerHandle)
		c.tokenizerHandle = nil
	}

	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// CreateChatCompletionStream creates a streaming chat completion using gRPC
// This uses preprocessing FFI for chat_template and tokenization,
// then direct gRPC calls, and batch postprocessing FFI for tool parsing
func (c *GrpcClient) CreateChatCompletionStream(ctx context.Context, reqJSON string) (*GrpcChatCompletionStream, error) {
	// Step 1: Preprocess using Rust FFI (chat_template + tokenization)
	// Tokenizer handle is pre-created at startup, so we can use it directly without locks
	// CRITICAL: TokenizerHandle is thread-safe (Arc<dyn TokenizerTrait> with Send + Sync)
	// All tokenizer methods are read-only (&self), so concurrent calls are safe
	// No lock needed - this eliminates lock contention and allows true parallelism
	if c.tokenizerHandle == nil {
		return nil, fmt.Errorf("tokenizer handle is nil (should be created at startup)")
	}

	// Direct call without lock - tokenizer is thread-safe
	// This allows multiple requests to preprocess concurrently, eliminating the bottleneck
	preprocessed, err := ffi.PreprocessChatRequestWithTokenizer(reqJSON, c.tokenizerHandle)
	if err != nil {
		return nil, fmt.Errorf("preprocessing failed: %w", err)
	}
	defer func() {
		if preprocessed != nil {
			preprocessed.Free()
		}
	}()

	// Parse request JSON to get parameters
	var reqMap map[string]interface{}
	if err := json.Unmarshal([]byte(reqJSON), &reqMap); err != nil {
		return nil, fmt.Errorf("failed to parse request JSON: %w", err)
	}

	model, _ := reqMap["model"].(string)
	if model == "" {
		model = "default"
	}

	// Build GenerateRequest
	generateReq := &proto.GenerateRequest{
		RequestId: fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Tokenized: &proto.TokenizedInput{
			OriginalText: preprocessed.PromptText,
			InputIds:     preprocessed.TokenIDs,
		},
		Stream: true,
	}

	// Set sampling parameters
	samplingParams := &proto.SamplingParams{
		Temperature:       1.0,
		TopP:              1.0,
		TopK:              -1,
		SkipSpecialTokens: true,
	}

	if temp, ok := reqMap["temperature"].(float64); ok {
		samplingParams.Temperature = float32(temp)
	}
	if topP, ok := reqMap["top_p"].(float64); ok {
		samplingParams.TopP = float32(topP)
	}
	if topK, ok := reqMap["top_k"].(float64); ok {
		samplingParams.TopK = int32(topK)
	}
	// CRITICAL: Check both max_completion_tokens and max_tokens (matching Rust behavior)
	// Priority: max_completion_tokens (recommended) > max_tokens (deprecated, backward compatibility)
	// Rust normalize() migrates max_tokens → max_completion_tokens, then only uses max_completion_tokens
	var maxTokensInt *int32
	if maxCompletionTokens, ok := reqMap["max_completion_tokens"].(float64); ok {
		// Use max_completion_tokens (recommended field, used by bench_serving.py)
		tokens := int32(maxCompletionTokens)
		maxTokensInt = &tokens
	} else if maxTokens, ok := reqMap["max_tokens"].(float64); ok {
		// Fallback to max_tokens (deprecated, for backward compatibility with OpenAI API)
		tokens := int32(maxTokens)
		maxTokensInt = &tokens
	}
	if maxTokensInt != nil {
		samplingParams.MaxNewTokens = maxTokensInt
	}

	// Parse tool constraints if available
	if preprocessed.ToolConstraintsJSON != "" {
		var toolConstraints map[string]interface{}
		if err := json.Unmarshal([]byte(preprocessed.ToolConstraintsJSON), &toolConstraints); err == nil {
			if regex, ok := toolConstraints["regex"].(string); ok {
				samplingParams.Constraint = &proto.SamplingParams_Regex{Regex: regex}
			} else if jsonSchema, ok := toolConstraints["json_schema"].(string); ok {
				samplingParams.Constraint = &proto.SamplingParams_JsonSchema{JsonSchema: jsonSchema}
			}
		}
	}

	generateReq.SamplingParams = samplingParams
	generateReq.Timestamp = timestamppb.Now()

	// Create gRPC stream
	// Note: Generate() sends the request and returns a stream for receiving responses
	stream, err := c.client.Generate(ctx, generateReq)
	if err != nil {
		return nil, fmt.Errorf("failed to create gRPC stream: %w", err)
	}

	// Note: We don't defer cleanup here because the stream needs to stay alive
	// The stream will be closed in GrpcChatCompletionStream.Close()

	// Create converter handle for postprocessing
	// We need to create it from the request JSON
	toolsJSON := ""
	if tools, ok := reqMap["tools"].([]interface{}); ok && len(tools) > 0 {
		toolsBytes, _ := json.Marshal(tools)
		toolsJSON = string(toolsBytes)
	}

	toolChoiceJSON := ""
	if toolChoice, ok := reqMap["tool_choice"]; ok {
		toolChoiceBytes, _ := json.Marshal(toolChoice)
		toolChoiceJSON = string(toolChoiceBytes)
	}

	stopJSON := ""
	if stop, ok := reqMap["stop"]; ok {
		stopBytes, _ := json.Marshal(stop)
		stopJSON = string(stopBytes)
	}

	// Create converter handle (this will be used for postprocessing)
	stopTokenIDs := []uint32{}
	if stopTokenIDsVal, ok := reqMap["stop_token_ids"].([]interface{}); ok {
		for _, id := range stopTokenIDsVal {
			if idFloat, ok := id.(float64); ok {
				stopTokenIDs = append(stopTokenIDs, uint32(idFloat))
			}
		}
	}

	skipSpecialTokens := true
	if skipSpecialTokensVal, ok := reqMap["skip_special_tokens"].(bool); ok {
		skipSpecialTokens = skipSpecialTokensVal
	}

	// Create converter handle for postprocessing
	// Tokenizer handle is pre-created at startup, so we can use it directly
	if c.tokenizerHandle == nil {
		stream.CloseSend()
		return nil, fmt.Errorf("tokenizer handle is nil (should be created at startup)")
	}

	// Use pre-created tokenizer to create converter (much faster!)
	converterHandle, err := ffi.CreateGrpcResponseConverterWithTokenizer(
		c.tokenizerHandle,
		model,
		generateReq.RequestId,
		toolsJSON,
		toolChoiceJSON,
		stopJSON,
		stopTokenIDs,
		skipSpecialTokens,
		preprocessed.PromptTokens, // Pass initial prompt tokens from preprocessing
	)
	if err != nil {
		stream.CloseSend()
		return nil, fmt.Errorf("failed to create converter handle: %w", err)
	}

	// Batch postprocessor for processing chunks
	// batchSize=1: Process immediately without batching delay
	batchSize := 1
	batchPostprocessor := ffi.NewBatchPostprocessor(converterHandle, batchSize, 0) // 0 = immediate processing

	grpcStream := &GrpcChatCompletionStream{
		stream:             stream,
		converterHandle:    converterHandle,
		batchPostprocessor: batchPostprocessor,
		batchSize:          batchSize,
		ctx:                ctx,
		resultJSONChan:     make(chan string, 2000), // Buffer for processed JSON responses
		errChan:            make(chan error, 100),   // Buffer for errors
		readLoopDone:       make(chan struct{}),
		requestID:          generateReq.RequestId,
		model:              model,
	}

	// Start async read loop to process stream in background
	// This eliminates blocking and reduces TPOT latency significantly
	go grpcStream.readLoop()

	return grpcStream, nil
}

// GrpcChatCompletionStream represents a streaming chat completion via gRPC
type GrpcChatCompletionStream struct {
	stream             grpcClientStream
	converterHandle    *ffi.GrpcResponseConverterHandle
	batchPostprocessor *ffi.BatchPostprocessor
	batchSize          int
	ctx                context.Context
	closed             int32         // Atomic flag for closed state
	resultJSONChan     chan string   // Channel for processed JSON responses (from FFI)
	errChan            chan error    // Channel for errors
	readLoopDone       chan struct{} // Signal when read loop is done
	requestID          string
	model              string
}

// readLoop runs in a background goroutine to continuously read from gRPC stream
// and process chunks asynchronously, eliminating blocking delays
func (s *GrpcChatCompletionStream) readLoop() {
	defer close(s.readLoopDone)
	defer close(s.resultJSONChan)

	// CRITICAL: Use a channel to make Recv() cancelable
	// Use a dedicated goroutine for Recv() to avoid goroutine leaks
	// Increased buffer to prevent blocking when readLoop is processing
	recvChan := make(chan recvResult, 1000) // Increased from 1 to 10 to prevent blocking
	recvDone := make(chan struct{})

	// Start a single dedicated goroutine for Recv() calls
	// CRITICAL: This goroutine will be unblocked when stream is closed (via CloseSend())
	go func() {
		defer close(recvDone)
		for {
			// Check if context is cancelled before calling Recv()
			select {
			case <-s.ctx.Done():
				return
			default:
			}

			// Check if stream is closed
			if atomic.LoadInt32(&s.closed) == 1 {
				return
			}

			// Call Recv() - this is blocking, but we're in a dedicated goroutine
			// When stream is closed (via CloseSend()), Recv() will return an error
			protoResp, err := s.stream.Recv()

			// If Recv() returned an error (including EOF or stream closed), exit the goroutine
			// This happens when:
			// 1. Stream ends normally (EOF)
			// 2. Stream is closed via CloseSend() (context cancellation or Close())
			// 3. Stream error
			if err != nil {
				// Try to send error result, but don't block if context is cancelled
				select {
				case recvChan <- recvResult{resp: protoResp, err: err}:
					// Successfully sent error result
				case <-s.ctx.Done():
					// Context cancelled while trying to send - exit
					return
				default:
					// recvChan is full and context not cancelled - exit anyway
					// This should be rare since recvChan has buffer size 1
					return
				}
				return
			}

			// Try to send result, but respect context cancellation
			select {
			case recvChan <- recvResult{resp: protoResp, err: err}:
				// Successfully sent result
			case <-s.ctx.Done():
				// Context cancelled while trying to send - exit
				return
			}
		}
	}()

	for {
		// Check if stream is closed
		if atomic.LoadInt32(&s.closed) == 1 {
			return
		}

		// Wait for Recv() result or context cancellation
		select {
		case <-s.ctx.Done():
			// CRITICAL: When context is cancelled, close the stream to unblock Recv()
			// This ensures the Recv() goroutine can exit
			_ = s.stream.CloseSend()
			return
		case result, ok := <-recvChan:
			if !ok {
				// recvChan closed - Recv() goroutine exited
				return
			}
			if result.err != nil {
				if result.err == io.EOF {
					// Stream ended - flush remaining chunks and exit
					results, flushErr := s.flushBatch()
					if flushErr != nil {
						select {
						case s.errChan <- fmt.Errorf("failed to flush batch: %w", flushErr):
						default:
						}
						return
					}
					// Send all flushed results as raw JSON
					for _, resultJSON := range results {
						select {
						case s.resultJSONChan <- resultJSON:
							// Successfully sent
						case <-s.ctx.Done():
							return
						}
					}
					return
				}
				// Send error to errChan
				select {
				case s.errChan <- result.err:
				default:
				}
				return
			}

			// Process and send to resultJSONChan
			if result.resp != nil {
				s.processAndSendResponse(result.resp)
			}
		}
	}
}

// processAndSendResponse processes proto response and sends to resultJSONChan
// This is called from the readLoop goroutine
func (s *GrpcChatCompletionStream) processAndSendResponse(protoResp *proto.GenerateResponse) {
	// Check if stream is closed
	if atomic.LoadInt32(&s.closed) == 1 {
		return
	}

	// Check if protoResp is nil
	if protoResp == nil {
		return
	}

	// Convert proto response to JSON for FFI postprocessing
	protoJSON, err := protoToJSON(protoResp)
	if err != nil {
		select {
		case s.errChan <- fmt.Errorf("failed to convert proto to JSON: %w", err):
		default:
		}
		return
	}

	// Use batch postprocessor to process chunk
	if s.batchPostprocessor == nil {
		select {
		case s.errChan <- fmt.Errorf("batch postprocessor is nil"):
		default:
		}
		return
	}

	results, _, err := s.batchPostprocessor.AddChunk(protoJSON)
	if err != nil {
		select {
		case s.errChan <- fmt.Errorf("batch postprocessing failed: %w", err):
		default:
		}
		return
	}

	// Send processed JSON strings to resultJSONChan
	// The JSON is already in OpenAI format from Rust FFI
	for _, resultJSON := range results {
		if atomic.LoadInt32(&s.closed) == 1 {
			return
		}
		select {
		case s.resultJSONChan <- resultJSON:
			// Successfully sent
		case <-s.ctx.Done():
			// Context cancelled - exit
			return
		}
	}
}

// Recv receives the next chunk from the stream
// Uses lazy parsing - JSON is parsed here instead of in readLoop to reduce blocking
func (s *GrpcChatCompletionStream) Recv() (*ChatCompletionStreamResponse, error) {
	select {
	case resultJSON, ok := <-s.resultJSONChan:
		if !ok {
			return nil, io.EOF
		}
		// Skip empty JSON
		if resultJSON == "" {
			return s.Recv()
		}
		// Parse JSON (lazy parsing - moved from readLoop)
		var response ChatCompletionStreamResponse
		if err := json.Unmarshal([]byte(resultJSON), &response); err != nil {
			return nil, fmt.Errorf("failed to parse response: %w", err)
		}
		return &response, nil
	case err, ok := <-s.errChan:
		if !ok {
			return nil, io.EOF
		}
		return nil, err
	case <-s.ctx.Done():
		return nil, s.ctx.Err()
	}
}

// RecvJSON receives the next chunk as raw JSON string
// This avoids JSON unmarshal/marshal overhead compared to Recv()
func (s *GrpcChatCompletionStream) RecvJSON() (string, error) {
	select {
	case resultJSON, ok := <-s.resultJSONChan:
		if !ok {
			return "", io.EOF
		}
		// Skip empty JSON
		if resultJSON == "" {
			return s.RecvJSON()
		}
		return resultJSON, nil
	case err, ok := <-s.errChan:
		if !ok {
			return "", io.EOF
		}
		return "", err
	case <-s.ctx.Done():
		return "", s.ctx.Err()
	}
}

// Close closes the stream
func (s *GrpcChatCompletionStream) Close() error {
	// Use atomic compare-and-swap to set closed flag
	if !atomic.CompareAndSwapInt32(&s.closed, 0, 1) {
		// Already closed
		return nil
	}

	// Wait for read loop to finish (with timeout)
	select {
	case <-s.readLoopDone:
	case <-time.After(5 * time.Second):
		// Timeout after 5 seconds
	}

	// Flush remaining chunks (ignore results on close)
	_, _ = s.flushBatch()

	// Close converter handle
	if s.converterHandle != nil {
		ffi.FreeGrpcResponseConverter(s.converterHandle)
	}

	// Close gRPC stream
	return s.stream.CloseSend()
}

// flushBatch flushes remaining chunks in the batch postprocessor
// Returns flushed results and any error
func (s *GrpcChatCompletionStream) flushBatch() ([]string, error) {
	if s.batchPostprocessor != nil {
		results, err := s.batchPostprocessor.Flush()
		if err != nil {
			return nil, fmt.Errorf("batch flush failed: %w", err)
		}
		return results, nil
	}
	return nil, nil
}

// protoToJSON converts a proto GenerateResponse to JSON string
// Optimized version: directly builds JSON string to avoid map allocation and extra marshaling
// Performance: Minimizes json.Marshal calls by manually formatting simple types
func protoToJSON(resp *proto.GenerateResponse) (string, error) {
	var sb strings.Builder
	// Pre-allocate capacity for typical chunk size (~300-500 bytes)
	sb.Grow(500)

	sb.WriteString(`{"request_id":`)
	// Only marshal request_id if it's not a simple string
	if resp.RequestId == "" {
		sb.WriteString(`""`)
	} else {
		requestIDJSON, err := json.Marshal(resp.RequestId)
		if err != nil {
			return "", err
		}
		sb.Write(requestIDJSON)
	}

	switch r := resp.Response.(type) {
	case *proto.GenerateResponse_Chunk:
		sb.WriteString(`,"chunk":{`)
		sb.WriteString(`"token_ids":`)
		// Optimize: Only marshal token_ids array (required for Rust processing)
		// Other fields are simple integers, format directly
		tokenIDsJSON, err := json.Marshal(r.Chunk.TokenIds)
		if err != nil {
			return "", err
		}
		sb.Write(tokenIDsJSON)
		sb.WriteString(`,"prompt_tokens":`)
		sb.WriteString(strconv.FormatInt(int64(r.Chunk.PromptTokens), 10))
		sb.WriteString(`,"completion_tokens":`)
		sb.WriteString(strconv.FormatInt(int64(r.Chunk.CompletionTokens), 10))
		sb.WriteString(`,"cached_tokens":`)
		sb.WriteString(strconv.FormatInt(int64(r.Chunk.CachedTokens), 10))
		sb.WriteString(`,"index":`)
		sb.WriteString(strconv.FormatInt(int64(r.Chunk.Index), 10))
		sb.WriteString(`}`)
	case *proto.GenerateResponse_Complete:
		sb.WriteString(`,"complete":{`)
		sb.WriteString(`"output_ids":`)
		outputIDsJSON, err := json.Marshal(r.Complete.OutputIds)
		if err != nil {
			return "", err
		}
		sb.Write(outputIDsJSON)
		sb.WriteString(`,"finish_reason":`)
		finishReasonJSON, err := json.Marshal(r.Complete.FinishReason)
		if err != nil {
			return "", err
		}
		sb.Write(finishReasonJSON)
		sb.WriteString(`,"prompt_tokens":`)
		sb.WriteString(strconv.FormatInt(int64(r.Complete.PromptTokens), 10))
		sb.WriteString(`,"completion_tokens":`)
		sb.WriteString(strconv.FormatInt(int64(r.Complete.CompletionTokens), 10))
		sb.WriteString(`,"cached_tokens":`)
		sb.WriteString(strconv.FormatInt(int64(r.Complete.CachedTokens), 10))
		sb.WriteString(`}`)
	case *proto.GenerateResponse_Error:
		sb.WriteString(`,"error":{`)
		sb.WriteString(`"message":`)
		messageJSON, err := json.Marshal(r.Error.Message)
		if err != nil {
			return "", err
		}
		sb.Write(messageJSON)
		sb.WriteString(`,"http_status_code":`)
		httpStatusCodeJSON, err := json.Marshal(r.Error.HttpStatusCode)
		if err != nil {
			return "", err
		}
		sb.Write(httpStatusCodeJSON)
		if r.Error.Details != "" {
			sb.WriteString(`,"details":`)
			detailsJSON, err := json.Marshal(r.Error.Details)
			if err != nil {
				return "", err
			}
			sb.Write(detailsJSON)
		}
		sb.WriteString(`}`)
	}

	sb.WriteString(`}`)
	return sb.String(), nil
}

// ChatCompletionStreamResponse represents a streaming chat completion response
// This must match the type in client.go for compatibility
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
