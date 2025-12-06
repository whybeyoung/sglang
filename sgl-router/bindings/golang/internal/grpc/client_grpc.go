// Package grpc provides gRPC client implementation for SGLang
package grpc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
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
	// Check if FFI preprocessing should be disabled (for performance testing)
	disableFFIPreprocess := os.Getenv("DISABLE_FFI_PREPROCESS") == "true"

	var preprocessed *ffi.PreprocessedRequest
	var err error

	if !disableFFIPreprocess {
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
		preprocessed, err = ffi.PreprocessChatRequestWithTokenizer(reqJSON, c.tokenizerHandle)

		if err != nil {
			return nil, fmt.Errorf("preprocessing failed: %w", err)
		}
		defer func() {
			if preprocessed != nil {
				preprocessed.Free()
			}
		}()
	} else {
		// Skip FFI preprocessing - use placeholder data for performance testing
		// Parse request JSON to extract basic info
		var reqMap map[string]interface{}
		if err := json.Unmarshal([]byte(reqJSON), &reqMap); err != nil {
			return nil, fmt.Errorf("failed to parse request JSON: %w", err)
		}

		// Extract messages and build simple prompt text (no chat_template)
		promptText := ""
		if messages, ok := reqMap["messages"].([]interface{}); ok {
			var parts []string
			for _, msg := range messages {
				if msgMap, ok := msg.(map[string]interface{}); ok {
					if role, ok := msgMap["role"].(string); ok {
						if content, ok := msgMap["content"].(string); ok {
							parts = append(parts, fmt.Sprintf("%s: %s", role, content))
						}
					}
				}
			}
			promptText = strings.Join(parts, "\n")
		}

		// Use placeholder token IDs for performance testing
		// In real scenario, we'd need tokenization, but we skip it to test pure gRPC performance
		// This will produce incorrect output, but allows us to measure gRPC overhead
		placeholderTokenIDs := make([]uint32, 100) // Placeholder: 100 tokens
		for i := range placeholderTokenIDs {
			placeholderTokenIDs[i] = uint32(i + 1)
		}

		// Create a mock preprocessed request (no C memory allocated, so no Free() needed)
		// The internal pointers (promptTextPtr, tokenIDsPtr, etc.) will be nil by default,
		// so Free() will correctly skip freeing C memory
		preprocessed = &ffi.PreprocessedRequest{
			PromptText:          promptText,
			TokenIDs:            placeholderTokenIDs,
			ToolConstraintsJSON: "",
			PromptTokens:        int32(len(placeholderTokenIDs)),
		}
	}

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
	if maxTokens, ok := reqMap["max_tokens"].(float64); ok {
		maxTokensInt := int32(maxTokens)
		samplingParams.MaxNewTokens = &maxTokensInt
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

	// Check if FFI postprocessing should be disabled (for performance testing)
	disableFFIPostprocess := os.Getenv("DISABLE_FFI_POSTPROCESS") == "true"

	var converterHandle *ffi.GrpcResponseConverterHandle
	var tokenizerHandle *ffi.TokenizerHandle
	var batchPostprocessor *ffi.BatchPostprocessor
	batchSize := 1 // Default batch size for optimization check

	// Only create converter handle if postprocessing is enabled
	if !disableFFIPostprocess {
		// Tokenizer handle is pre-created at startup, so we can use it directly
		if c.tokenizerHandle == nil {
			stream.CloseSend()
			return nil, fmt.Errorf("tokenizer handle is nil (should be created at startup)")
		}

		// Use pre-created tokenizer to create converter (much faster!)
		converterHandle, err = ffi.CreateGrpcResponseConverterWithTokenizer(
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

		// CRITICAL: Optimize for ultra-low latency TPOT (target: ~28ms like Rust)
		// - batchSize=1: No batching delay - process immediately
		// - flushInterval=0: No timeout - process immediately when batch is full
		// Key insight: Any batching delay directly adds to TPOT
		// Trade-off: More FFI calls (1 per token) but eliminates ALL batching delay
		// This matches Rust's per-token processing approach
		batchSize = 1
		batchPostprocessor = ffi.NewBatchPostprocessor(converterHandle, batchSize, 0) // 0 = immediate processing
	} else {
		// If postprocessing is disabled, we still need tokenizer handle for direct decoding (if enabled later)
		// But for pure gRPC testing, we can skip it
		// For now, set to nil - will be created on demand if needed
		tokenizerHandle = nil
	}

	grpcStream := &GrpcChatCompletionStream{
		stream:                stream,
		converterHandle:       converterHandle,
		batchPostprocessor:    batchPostprocessor,
		batchSize:             batchSize, // Store for optimization check
		ctx:                   ctx,
		resultChan:            make(chan *ChatCompletionStreamResponse, 2000), // Increased buffer to prevent channel full deadlock
		resultJSONChan:        make(chan string, 2000),                        // Increased buffer to prevent channel full deadlock
		errChan:               make(chan error, 100),                          // Larger error buffer for concurrent requests
		readLoopDone:          make(chan struct{}),
		disableFFIPostprocess: disableFFIPostprocess,
		disableFFIPreprocess:  disableFFIPreprocess, // Store for debug logging
		requestID:             generateReq.RequestId,
		model:                 model,
		tokenizerHandle:       tokenizerHandle, // Cache tokenizer for direct decoding when FFI disabled
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
	batchSize          int // Store batchSize for optimization check
	ctx                context.Context
	// Removed mu sync.Mutex - no longer needed, using atomic for closed flag
	closed int32 // Use atomic for closed flag to reduce lock contention
	// Removed pendingResults - no longer used, replaced by channel-based approach
	resultChan            chan *ChatCompletionStreamResponse // Channel for async results (parsed, for disableFFIPostprocess)
	resultJSONChan        chan string                        // Channel for raw JSON strings (lazy parsing to avoid blocking readLoop)
	errChan               chan error                         // Channel for errors
	readLoopDone          chan struct{}                      // Signal when read loop is done
	disableFFIPostprocess bool                               // Skip FFI postprocessing for performance testing
	disableFFIPreprocess  bool                               // Skip FFI preprocessing (for debug logging)
	requestID             string                             // Request ID for responses
	model                 string                             // Model name for responses
	tokenizerHandle       *ffi.TokenizerHandle               // Cached tokenizer for direct decoding when FFI disabled
}

// readLoop runs in a background goroutine to continuously read from gRPC stream
// and process chunks asynchronously, eliminating blocking delays
func (s *GrpcChatCompletionStream) readLoop() {
	defer close(s.readLoopDone)

	// Track TPOT metrics for gRPC responses
	var firstTokenTime time.Time
	var tokenCount int

	// CRITICAL: Use goroutine + channel to handle blocking Recv()
	// Similar to Rust's async stream.next().await pattern
	// This allows us to check context cancellation while Recv() is blocking
	// Buffer size 10 allows goroutine to send multiple responses even if main loop is slow
	grpcRespChan := make(chan struct {
		resp *proto.GenerateResponse
		err  error
	}, 2048)

	// Start goroutine to read from gRPC stream (blocking operation)
	// CRITICAL: Use goroutine + channel to handle blocking Recv()
	// Similar to Rust's async stream.next().await pattern
	// This allows us to check context cancellation while Recv() is blocking
	// Rust doesn't use explicit timeout, it relies on async runtime and context cancellation
	recvGoroutineDone := make(chan struct{})
	go func() {
		defer close(grpcRespChan)
		defer close(recvGoroutineDone)
		for {
			// Check context before Recv()
			select {
			case <-s.ctx.Done():
				return
			default:
			}

			// CRITICAL: Recv() is blocking - this goroutine will block here
			// Similar to Rust's stream.next().await, which blocks but can be cancelled via context
			// We rely on gRPC stream's context cancellation to handle this
			protoResp, err := s.stream.Recv()

			// Send result to channel
			select {
			case grpcRespChan <- struct {
				resp *proto.GenerateResponse
				err  error
			}{protoResp, err}:
				// Successfully sent
			case <-s.ctx.Done():
				return
			}

			// If error or EOF, exit goroutine
			if err != nil {
				return
			}
		}
	}()

	// CRITICAL: Use separate goroutine to send to resultJSONChan to avoid deadlock
	// Main loop reads from grpcRespChan, processing goroutine sends to resultJSONChan
	// This prevents main loop from blocking when resultJSONChan is full
	processChan := make(chan struct {
		protoResp *proto.GenerateResponse
		err       error
	}, 100) // Buffer to allow main loop to continue reading

	// Processing goroutine: handles data conversion and sends to resultJSONChan
	// This goroutine is responsible for closing resultChan and resultJSONChan (where it sends)
	go func() {
		defer close(s.resultChan)
		defer close(s.resultJSONChan)
		for {
			select {
			case <-s.ctx.Done():
				return
			case item, ok := <-processChan:
				if !ok {
					// processChan closed by main loop, exit and close result channels
					return
				}
				if item.err != nil {
					// Error already handled in main loop
					continue
				}
				if item.protoResp == nil {
					continue
				}

				// Process and send to resultJSONChan (may block, but doesn't affect main loop)
				s.processAndSendResponse(item.protoResp)
			}
		}
	}()

	for {
		// Check if stream is closed (use atomic read to avoid lock contention)
		if atomic.LoadInt32(&s.closed) == 1 {
			// Close processChan to signal processing goroutine to exit
			close(processChan)
			return
		}

		// Wait for Recv() result or context cancellation
		var protoResp *proto.GenerateResponse
		var err error

		select {
		case <-s.ctx.Done():
			// Close processChan to signal processing goroutine to exit
			close(processChan)
			return
		case result, ok := <-grpcRespChan:
			if !ok {
				// Channel closed - goroutine exited
				// Close processChan to signal processing goroutine to exit
				close(processChan)
				return
			}
			protoResp = result.resp
			err = result.err
		}

		if err != nil {
			if err == io.EOF {
				// Stream ended - flush remaining chunks and exit immediately
				if !s.disableFFIPostprocess {
					// Flush remaining chunks - this is critical for getting the final complete message with usage
					results, flushErr := s.flushBatch()
					if flushErr != nil {
						select {
						case s.errChan <- fmt.Errorf("failed to flush batch: %w", flushErr):
						default:
						}
						// Close processChan to signal processing goroutine to exit
						close(processChan)
						return
					}
					// Send all flushed results as raw JSON (parse in Recv())
					// CRITICAL: Must wait for channel space to avoid deadlock
					// If we skip sending when channel is full, Recv() will wait forever
					for _, resultJSON := range results {
						select {
						case s.resultJSONChan <- resultJSON:
							// Successfully sent
						case <-s.ctx.Done():
							// Close processChan to signal processing goroutine to exit
							close(processChan)
							return
							// Removed default case - must wait for channel space to prevent deadlock
							// If channel is consistently full, it means Recv() is not consuming fast enough
							// In that case, we should wait rather than skip, to maintain data flow
						}
					}
				}
				// Log TPOT metrics before exit
				if tokenCount > 1 {
					avgTPOT := time.Since(firstTokenTime) / time.Duration(tokenCount-1)
					fmt.Printf("[GRPC_TPOT] request_id=%s tokens=%d avg_tpot=%v\n", s.requestID, tokenCount, avgTPOT)
				}
				// CRITICAL: Exit immediately after EOF - don't continue loop
				// This prevents readLoop from calling Recv() again after stream is closed
				// If we continue, Recv() may block indefinitely waiting for data that will never arrive
				// Close processChan to signal processing goroutine to exit
				close(processChan)
				return
			}
			select {
			case s.errChan <- err:
			default:
			}
			// Close processChan to signal processing goroutine to exit
			close(processChan)
			return
		}

		// Track TPOT: record time when we receive each response from gRPC
		if err == nil && protoResp != nil {
			if tokenCount == 0 {
				firstTokenTime = time.Now()
			}
			tokenCount++

			// Send to processing goroutine (non-blocking, allows main loop to continue)
			select {
			case processChan <- struct {
				protoResp *proto.GenerateResponse
				err       error
			}{protoResp, err}:
				// Successfully sent to processing goroutine
			case <-s.ctx.Done():
				return
			default:
				// Processing goroutine is slow, but we continue reading from grpcRespChan
				// This prevents deadlock: main loop can continue even if processing is slow
				// Note: This may cause some data to be dropped, but prevents deadlock
				select {
				case s.errChan <- fmt.Errorf("processChan full, dropping response"):
				default:
				}
			}
		}

		// Continue loop to read next response (don't process here to avoid blocking)
		continue
	}
}

// processAndSendResponse processes proto response and sends to resultJSONChan
// This is called from processing goroutine, so blocking here doesn't affect main loop
func (s *GrpcChatCompletionStream) processAndSendResponse(protoResp *proto.GenerateResponse) {
	// Check if context is cancelled or stream is closed
	select {
	case <-s.ctx.Done():
		return
	default:
	}
	if atomic.LoadInt32(&s.closed) == 1 {
		return
	}

	// Check if protoResp is nil (shouldn't happen, but be defensive)
	if protoResp == nil {
		return
	}

	// Convert proto response directly to OpenAI format (skip FFI postprocessing if disabled)
	if s.disableFFIPostprocess {
		// Direct conversion without FFI - for performance testing
		response := protoToOpenAIResponse(protoResp, s.requestID, s.model, s.tokenizerHandle)
		if response == nil {
			// This shouldn't happen anymore since protoToOpenAIResponse now always returns a response
			// But keep this check for safety
			return
		}

		// Send response to channel
		// CRITICAL: Must wait for channel space to avoid deadlock
		// If we skip sending when channel is full, Recv() will wait forever
		// Check context and closed flag before sending
		select {
		case <-s.ctx.Done():
			return
		default:
		}
		if atomic.LoadInt32(&s.closed) == 1 {
			return
		}
		select {
		case s.resultChan <- response:
			// Successfully sent to channel
		case <-s.ctx.Done():
			return
			// Removed default case - must wait for channel space to prevent deadlock
			// If channel is consistently full, it means Recv() is not consuming fast enough
			// In that case, we should wait rather than skip, to maintain data flow
		}
	} else {
		// Use FFI postprocessing (normal path)
		// Convert proto response to JSON for postprocessing
		protoJSON, err := protoToJSON(protoResp)

		if err != nil {
			select {
			case s.errChan <- fmt.Errorf("failed to convert proto to JSON: %w", err):
			default:
			}
			return
		}

		// Use batch postprocessor for all cases (batchSize > 1)
		// This reduces FFI call frequency and improves throughput
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

		// CRITICAL: Send raw JSON strings to avoid parsing in readLoop
		// Parse JSON only when Recv() is called (lazy parsing)
		// This reduces readLoop blocking time and improves TPOT significantly
		// The JSON string is already in OpenAI format from Rust FFI
		// CRITICAL: Must wait for channel space to avoid deadlock
		// If we skip sending when channel is full, Recv() will wait forever
		if len(results) > 0 {
			for _, resultJSON := range results {
				// Check context and closed flag before sending
				select {
				case <-s.ctx.Done():
					return
				default:
				}
				if atomic.LoadInt32(&s.closed) == 1 {
					return
				}
				// CRITICAL: Must wait for channel space, cannot skip
				// If we skip, Recv() will wait forever for data that never arrives
				// This causes deadlock: readLoop continues but doesn't send, Recv() waits forever
				select {
				case s.resultJSONChan <- resultJSON:
					// Successfully sent raw JSON - parsing deferred to Recv()
				case <-s.ctx.Done():
					return
					// Removed default case - must wait for channel space to prevent deadlock
					// If channel is consistently full, it means Recv() is not consuming fast enough
					// In that case, we should wait rather than skip, to maintain data flow
				}
			}
		}
	}
}

// Recv receives the next chunk from the stream
// Now uses async channel-based approach for low latency
// CRITICAL: Parse JSON here (lazy parsing) to avoid blocking readLoop
func (s *GrpcChatCompletionStream) Recv() (*ChatCompletionStreamResponse, error) {
	// Always block waiting for data - no default case to avoid busy-waiting
	// This ensures proper synchronization and avoids CPU spinning
	select {
	case resultJSON, ok := <-s.resultJSONChan:
		// Channel closed - stream ended
		if !ok {
			return nil, io.EOF
		}
		// Empty string means invalid JSON - skip it
		if resultJSON == "" {
			// Try next item or return EOF
			return s.Recv()
		}
		// Parse JSON here (lazy parsing) - moved from readLoop to reduce blocking
		var response ChatCompletionStreamResponse
		if err := json.Unmarshal([]byte(resultJSON), &response); err != nil {
			return nil, fmt.Errorf("failed to parse response: %w", err)
		}

		return &response, nil
	case result, ok := <-s.resultChan:
		// Channel closed - stream ended
		if !ok {
			return nil, io.EOF
		}
		// Fallback: direct response (for disableFFIPostprocess path)
		if result == nil {
			return nil, io.EOF
		}
		return result, nil
	case err, ok := <-s.errChan:
		// Channel closed - stream ended
		if !ok {
			return nil, io.EOF
		}
		return nil, err
	case <-s.ctx.Done():
		return nil, s.ctx.Err()
	}
}

// RecvJSON receives the next chunk as raw JSON string (optimized path to avoid parsing/serialization)
// This is much faster than Recv() because it avoids JSON unmarshal/marshal overhead
func (s *GrpcChatCompletionStream) RecvJSON() (string, error) {
	// Always block waiting for data - no default case to avoid busy-waiting
	select {
	case resultJSON, ok := <-s.resultJSONChan:
		// Channel closed - stream ended
		if !ok {
			return "", io.EOF
		}
		// Empty string means invalid JSON - skip it
		if resultJSON == "" {
			// Try next item or return EOF
			return s.RecvJSON()
		}
		// Return raw JSON string directly - no parsing needed!
		return resultJSON, nil
	case result, ok := <-s.resultChan:
		// Channel closed - stream ended
		if !ok {
			return "", io.EOF
		}
		// Fallback: direct response (for disableFFIPostprocess path) - need to serialize
		if result == nil {
			return "", io.EOF
		}
		// Serialize to JSON (only for disableFFIPostprocess path)
		jsonData, err := json.Marshal(result)
		if err != nil {
			return "", fmt.Errorf("failed to marshal response: %w", err)
		}
		return string(jsonData), nil
	case err, ok := <-s.errChan:
		// Channel closed - stream ended
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

// protoToOpenAIResponse converts proto.GenerateResponse directly to ChatCompletionStreamResponse
// This bypasses FFI postprocessing for performance testing
// Note: This is a simplified conversion and may not handle all cases (e.g., tool calls, reasoning)
// For performance testing, we skip token decoding and use placeholder content
func protoToOpenAIResponse(resp *proto.GenerateResponse, requestID, model string, tokenizerHandle *ffi.TokenizerHandle) *ChatCompletionStreamResponse {
	response := &ChatCompletionStreamResponse{
		ID:      requestID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []StreamChoice{},
		Usage:   nil,
	}

	switch r := resp.Response.(type) {
	case *proto.GenerateResponse_Chunk:
		// For streaming chunks, we skip token decoding for performance testing
		// Return token IDs as JSON array string directly
		content := ""
		if len(r.Chunk.TokenIds) > 0 {
			// Convert token IDs to JSON array string
			tokenIDsJSON, err := json.Marshal(r.Chunk.TokenIds)
			if err == nil {
				content = string(tokenIDsJSON)
			}
		}

		response.Choices = []StreamChoice{
			{
				Index:        int(r.Chunk.Index),
				FinishReason: "",
				Delta: MessageDelta{
					Role:    "assistant",
					Content: content,
				},
			},
		}

		// Set usage if available
		if r.Chunk.PromptTokens > 0 || r.Chunk.CompletionTokens > 0 {
			response.Usage = &Usage{
				PromptTokens:     int(r.Chunk.PromptTokens),
				CompletionTokens: int(r.Chunk.CompletionTokens),
				TotalTokens:      int(r.Chunk.PromptTokens + r.Chunk.CompletionTokens),
			}
		}

	case *proto.GenerateResponse_Complete:
		// For complete message, set finish_reason
		finishReason := r.Complete.FinishReason
		if finishReason == "" {
			finishReason = "stop"
		}

		response.Object = "chat.completion.chunk"
		response.Choices = []StreamChoice{
			{
				Index:        0,
				FinishReason: finishReason,
				Delta: MessageDelta{
					Role:    "assistant",
					Content: "", // Complete message doesn't have delta content
				},
			},
		}

		// Set usage
		if r.Complete.PromptTokens > 0 || r.Complete.CompletionTokens > 0 {
			response.Usage = &Usage{
				PromptTokens:     int(r.Complete.PromptTokens),
				CompletionTokens: int(r.Complete.CompletionTokens),
				TotalTokens:      int(r.Complete.PromptTokens + r.Complete.CompletionTokens),
			}
		}

	case *proto.GenerateResponse_Error:
		// Error handling - create error response instead of returning nil
		// This ensures the error is properly sent to the client
		// Create an error response that will be sent to the client
		// The client can check for error in the response
		response.Object = "error"
		response.Choices = []StreamChoice{
			{
				Index:        0,
				FinishReason: "error",
				Delta: MessageDelta{
					Role:    "assistant",
					Content: "", // Error doesn't have content
				},
			},
		}
		// Note: We still return the response, but the client should check for error
		// For now, we'll send it and let the client handle it
		return response
	}

	return response
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
