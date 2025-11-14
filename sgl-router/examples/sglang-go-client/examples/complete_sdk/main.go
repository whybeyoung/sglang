// Complete SDK example demonstrating the full request-response flow using Rust FFI
package main

/*
#cgo LDFLAGS: -lsglang_router_rs -ldl
#include <stdlib.h>
#include <stdint.h>

// Forward declarations for Rust FFI functions
typedef int32_t SglErrorCode;
typedef void* SglangClientHandle;
typedef void* SglangStreamHandle;

SglangClientHandle* sgl_client_create(const char* endpoint, const char* tokenizer_path, char** error_out);
void sgl_client_free(SglangClientHandle* handle);
SglErrorCode sgl_client_chat_completion_stream(SglangClientHandle* client_handle, const char* request_json, SglangStreamHandle** stream_handle_out, char** error_out);
SglErrorCode sgl_stream_read_next(SglangStreamHandle* stream_handle, char** response_json_out, int* is_done_out, char** error_out);
void sgl_stream_free(SglangStreamHandle* handle);
void sgl_free_string(char* s);
*/
import "C"

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"
	"unsafe"
)

// SglErrorCode represents error codes from Rust FFI
type SglErrorCode int32

const (
	SglErrorSuccess         SglErrorCode = 0
	SglErrorInvalidArgument SglErrorCode = 1
	SglErrorTokenization    SglErrorCode = 2
	SglErrorParsing         SglErrorCode = 3
	SglErrorMemory          SglErrorCode = 4
	SglErrorUnknown         SglErrorCode = 99
)

// Helper function to get error message
func getErrorMessage(errorPtr **C.char) string {
	if errorPtr == nil || *errorPtr == nil {
		return ""
	}
	return C.GoString(*errorPtr)
}

// Helper function to free error message
func freeErrorMessage(errorPtr **C.char) {
	if errorPtr != nil && *errorPtr != nil {
		C.sgl_free_string(*errorPtr)
		*errorPtr = nil
	}
}

func main() {
	// Parse command line flags
	var (
		tokenizerPath = flag.String("tokenizer", "", "Path to tokenizer directory (required)")
		grpcEndpoint  = flag.String("endpoint", "", "gRPC endpoint address, e.g., grpc://localhost:20000 (required)")
		model         = flag.String("model", "default", "Model name")
		help          = flag.Bool("help", false, "Show help message")
	)
	flag.Parse()

	if *help {
		showHelp()
		os.Exit(0)
	}

	// Validate required parameters
	if *tokenizerPath == "" {
		log.Fatal("Error: tokenizer path is required. Use -tokenizer flag")
	}
	if *grpcEndpoint == "" {
		log.Fatal("Error: gRPC endpoint is required. Use -endpoint flag")
	}

	fmt.Println("=== Complete SDK Example ===")
	fmt.Printf("Tokenizer: %s\n", *tokenizerPath)
	fmt.Printf("Endpoint: %s\n", *grpcEndpoint)
	fmt.Printf("Model: %s\n", *model)
	fmt.Println()

	// Step 1: Create client handle
	fmt.Println("Step 1: Creating client handle...")
	var errorPtr *C.char
	cEndpoint := C.CString(*grpcEndpoint)
	cTokenizerPath := C.CString(*tokenizerPath)
	defer C.free(unsafe.Pointer(cEndpoint))
	defer C.free(unsafe.Pointer(cTokenizerPath))

	clientHandle := C.sgl_client_create(
		cEndpoint,
		cTokenizerPath,
		&errorPtr,
	)
	if clientHandle == nil {
		errorMsg := getErrorMessage(&errorPtr)
		freeErrorMessage(&errorPtr)
		log.Fatalf("Failed to create client: %s", errorMsg)
	}
	freeErrorMessage(&errorPtr)
	fmt.Println("✓ Client handle created successfully")
	defer func() {
		fmt.Println("\nCleaning up client handle...")
		C.sgl_client_free(clientHandle)
	}()

	// Step 2: Prepare OpenAI format request
	fmt.Println("\nStep 2: Preparing request...")
	request := map[string]interface{}{
		"model": *model,
		"messages": []map[string]interface{}{
			{
				"role":    "system",
				"content": "You are a helpful assistant.",
			},
			{
				"role":    "user",
				"content": "写一首关于春的诗歌.",
			},
		},
		"stream":                true,
		"temperature":           0.7,
		"max_completion_tokens": 500, // Use max_completion_tokens instead of max_tokens
		"skip_special_tokens":   true,
		"tools":                 []interface{}{}, // Provide empty array to avoid None in template
	}

	requestJSON, err := json.Marshal(request)
	if err != nil {
		log.Fatalf("Failed to marshal request: %v", err)
	}
	fmt.Printf("✓ Request prepared: %s\n", string(requestJSON))

	// Step 3: Send request and start streaming
	fmt.Println("\nStep 3: Sending request and starting stream...")
	var streamHandle *C.SglangStreamHandle
	errorPtr = nil
	cRequestJSON := C.CString(string(requestJSON))
	defer C.free(unsafe.Pointer(cRequestJSON))

	result := C.sgl_client_chat_completion_stream(
		clientHandle,
		cRequestJSON,
		&streamHandle,
		&errorPtr,
	)
	if SglErrorCode(result) != SglErrorSuccess {
		errorMsg := getErrorMessage(&errorPtr)
		freeErrorMessage(&errorPtr)
		log.Fatalf("Failed to start stream: error code %d, %s", result, errorMsg)
	}
	freeErrorMessage(&errorPtr)
	if streamHandle == nil {
		log.Fatal("Stream handle is null")
	}
	fmt.Println("✓ Stream started successfully")
	defer func() {
		fmt.Println("\nCleaning up stream handle...")
		C.sgl_stream_free(streamHandle)
	}()

	// Step 4: Read and process streaming responses
	fmt.Println("\nStep 4: Reading streaming responses...")
	fmt.Println("--- Response Stream ---")

	var fullContent strings.Builder
	chunkCount := 0
	startTime := time.Now()
	var firstTokenTime time.Time
	firstTokenReceived := false

	for {
		var responseJSON *C.char
		var isDone C.int
		errorPtr = nil

		result := C.sgl_stream_read_next(
			streamHandle,
			&responseJSON,
			&isDone,
			&errorPtr,
		)

		if SglErrorCode(result) != SglErrorSuccess {
			errorMsg := getErrorMessage(&errorPtr)
			freeErrorMessage(&errorPtr)
			if isDone == 1 {
				// Stream ended with error
				fmt.Printf("\n⚠ Stream ended with error: %s\n", errorMsg)
				break
			}
			log.Printf("Error reading stream: %s", errorMsg)
			continue
		}
		freeErrorMessage(&errorPtr)

		if isDone == 1 {
			fmt.Println("\n✓ Stream completed")
			break
		}

		if responseJSON == nil {
			continue
		}

		// Parse OpenAI format response
		responseStr := C.GoString(responseJSON)
		C.sgl_free_string(responseJSON)

		var response map[string]interface{}
		if err := json.Unmarshal([]byte(responseStr), &response); err != nil {
			log.Printf("Failed to parse response JSON: %v", err)
			continue
		}

		chunkCount++

		// Extract content delta
		if choices, ok := response["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				if delta, ok := choice["delta"].(map[string]interface{}); ok {
					if content, ok := delta["content"].(string); ok && content != "" {
						fmt.Print(content)
						fullContent.WriteString(content)

						// Track first token time (TTFT)
						if !firstTokenReceived {
							firstTokenTime = time.Now()
							firstTokenReceived = true
							ttft := firstTokenTime.Sub(startTime)
							fmt.Printf("\n[TTFT: %v]\n", ttft)
						}
					}
				}
			}
		}
	}

	// Calculate metrics
	if firstTokenReceived {
		elapsed := time.Since(startTime)
		tokensPerSecond := float64(fullContent.Len()) / elapsed.Seconds()
		fmt.Printf("\n\n--- Metrics ---\n")
		fmt.Printf("Total chunks: %d\n", chunkCount)
		fmt.Printf("Total content length: %d characters\n", fullContent.Len())
		fmt.Printf("Time elapsed: %v\n", elapsed)
		fmt.Printf("Tokens per second: %.2f\n", tokensPerSecond)
	}

	fmt.Println("\n=== Example completed ===")
}

func showHelp() {
	fmt.Println("Complete SDK Example - Full Request-Response Flow")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  go run main.go -tokenizer <path> -endpoint <endpoint> [options]")
	fmt.Println()
	fmt.Println("Options:")
	fmt.Println("  -tokenizer <path>    Path to tokenizer directory (required)")
	fmt.Println("  -endpoint <endpoint> gRPC endpoint, e.g., grpc://localhost:20000 (required)")
	fmt.Println("  -model <name>        Model name (default: 'default')")
	fmt.Println("  -help                Show this help message")
	fmt.Println()
	fmt.Println("Environment Variables:")
	fmt.Println("  SGL_TOKENIZER_PATH  Default tokenizer path")
	fmt.Println("  SGL_GRPC_ENDPOINT    Default gRPC endpoint")
	fmt.Println("  CARGO_BUILD_DIR      Cargo build directory (default: /Users/yangyanbo/cargobuild)")
}
