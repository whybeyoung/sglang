// Streaming tool call example demonstrating incremental tool call parsing
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/sglang/sgl-router-go-client/config"
	"github.com/sglang/sgl-router-go-client/internal/client"
	"github.com/sglang/sgl-router-go-client/internal/ffi"
	"github.com/sglang/sgl-router-go-client/internal/grpc"
	"github.com/sglang/sgl-router-go-client/internal/tokenizer"
)

func main() {
	// Parse command line flags
	var (
		tokenizerPath = flag.String("tokenizer", "", "Path to tokenizer directory or tokenizer.json file (required). "+
			"Recommended: specify directory to auto-discover tokenizer files")
		grpcEndpoint = flag.String("endpoint", "", "gRPC endpoint address, e.g., grpc://localhost:20000 (required)")
		parserType   = flag.String("parser", "auto", "Tool parser type: auto, json, llama, mistral, step3, etc.")
		help         = flag.Bool("help", false, "Show help message")
	)
	flag.Parse()

	if *help {
		showHelp()
		os.Exit(0)
	}

	// Load configuration
	var cfg *config.Config
	var err error

	// Override with command line arguments
	if *tokenizerPath != "" {
		if cfg == nil {
			cfg = config.DefaultConfig()
		}
		cfg.Tokenizer.Path = *tokenizerPath
	}
	if *grpcEndpoint != "" {
		if cfg == nil {
			cfg = config.DefaultConfig()
		}
		cfg.GRPC.Endpoint = *grpcEndpoint
	}

	// Validate required parameters
	if cfg == nil || cfg.Tokenizer.Path == "" {
		log.Fatal("Error: tokenizer path is required. Use -tokenizer flag or set SGL_TOKENIZER_PATH environment variable")
	}
	if cfg.GRPC.Endpoint == "" {
		log.Fatal("Error: gRPC endpoint is required. Use -endpoint flag or set SGL_GRPC_ENDPOINT environment variable")
	}

	// Create client
	clientConfig := client.Config{
		TokenizerPath: cfg.Tokenizer.Path,
		GRPCEndpoint:  cfg.GRPC.Endpoint,
		Timeout:       30 * time.Second,
	}

	cl, err := client.NewClient(clientConfig)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer cl.Close()

	// Define tools (functions)
	tools := []tokenizer.Tool{
		{
			Type: "function",
			Function: tokenizer.Function{
				Name:        "get_weather",
				Description: "Get the current weather in a given location",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "The city and state, e.g. San Francisco, CA",
						},
						"unit": map[string]interface{}{
							"type":        "string",
							"enum":        []string{"celsius", "fahrenheit"},
							"description": "The unit of temperature",
						},
					},
					"required": []string{"location"},
				},
			},
		},
		{
			Type: "function",
			Function: tokenizer.Function{
				Name:        "get_time",
				Description: "Get the current time in a given timezone",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"timezone": map[string]interface{}{
							"type":        "string",
							"description": "The timezone, e.g. America/New_York",
						},
					},
					"required": []string{"timezone"},
				},
			},
		},
	}

	// Create messages with tools
	ctx := context.Background()
	messages := []tokenizer.Message{
		{Role: "system", Content: "You are a helpful assistant with access to tools."},
		{Role: "user", Content: "What's the weather like in San Francisco?"},
	}

	// Create tool parser for incremental parsing
	parser, err := ffi.NewToolParser(*parserType)
	if err != nil {
		log.Fatalf("Failed to create tool parser: %v", err)
	}
	defer parser.Close()

	// Serialize tools to JSON for parser
	toolsJSON, err := json.Marshal(tools)
	if err != nil {
		log.Fatalf("Failed to marshal tools: %v", err)
	}
	toolsJSONStr := string(toolsJSON)

	// Build generation options
	options := grpc.DefaultGenerationOptions()
	maxTokens := int32(200)
	options.MaxNewTokens = &maxTokens
	options.SkipSpecialTokens = false // Important: don't skip special tokens for tool calls
	options.Stream = true

	// Start streaming generation
	fmt.Println("Starting streaming generation with tool calls...")
	fmt.Println("---")

	chunkChan, err := cl.GenerateStreamWithTools(ctx, messages, tools, options)
	if err != nil {
		log.Fatalf("Failed to start streaming: %v", err)
	}

	// Track metrics
	startTime := time.Now()
	var firstTokenTime time.Time
	firstTokenReceived := false
	var totalTokens int
	var accumulatedText strings.Builder
	var allToolCalls []ffi.ToolCall

	// Process stream chunks
	for chunk := range chunkChan {
		if chunk.Error != nil {
			log.Fatalf("Stream error: %v", chunk.Error)
		}

		// Record first token time
		if !firstTokenReceived && len(chunk.TokenIds) > 0 {
			firstTokenTime = time.Now()
			firstTokenReceived = true
			fmt.Printf("✓ First token received (TTFT: %v)\n", firstTokenTime.Sub(startTime))
		}

		// Accumulate text
		if chunk.Text != "" {
			accumulatedText.WriteString(chunk.Text)
			totalTokens += len(chunk.TokenIds)

			// Debug: Print raw chunk text (for debugging parser issues)
			if len(chunk.Text) > 0 {
				fmt.Printf("🔍 Raw chunk: %q\n", chunk.Text)
			}

			// Parse tool calls incrementally using Rust FFI
			result, err := parser.ParseIncremental(chunk.Text, toolsJSONStr)
			if err != nil {
				// Log error but continue
				log.Printf("Warning: Failed to parse incremental: %v", err)
			} else {
				// Print normal text if any
				if result.NormalText != "" {
					fmt.Printf("📝 Normal text: %s\n", result.NormalText)
				}

				// Print tool calls if any
				if len(result.ToolCalls) > 0 {
					for _, toolCall := range result.ToolCalls {
						// Check if this is a new tool call (not already seen)
						isNew := true
						for _, existing := range allToolCalls {
							if existing.ID == toolCall.ID {
								isNew = false
								break
							}
						}

						if isNew {
							allToolCalls = append(allToolCalls, toolCall)
							fmt.Printf("\n🔧 New Tool Call Detected:\n")
							fmt.Printf("   ID: %s\n", toolCall.ID)
							fmt.Printf("   Type: %s\n", toolCall.Type)
							fmt.Printf("   Function: %s\n", toolCall.Function.Name)
							if toolCall.Function.Arguments != "" {
								fmt.Printf("   Arguments: %s\n", toolCall.Function.Arguments)
							}
						} else {
							// Update existing tool call arguments
							for i := range allToolCalls {
								if allToolCalls[i].ID == toolCall.ID {
									if toolCall.Function.Arguments != "" {
										allToolCalls[i].Function.Arguments = toolCall.Function.Arguments
										fmt.Printf("\n🔧 Tool Call Updated:\n")
										fmt.Printf("   ID: %s\n", toolCall.ID)
										fmt.Printf("   Arguments: %s\n", toolCall.Function.Arguments)
									}
									break
								}
							}
						}
					}
				}
			}
		}

		// Handle completion
		if chunk.IsComplete {
			endTime := time.Now()
			totalDuration := endTime.Sub(startTime)
			var ttft time.Duration
			if firstTokenReceived {
				ttft = firstTokenTime.Sub(startTime)
			}

			fmt.Println("\n---")
			fmt.Println("✓ Generation complete!")
			fmt.Printf("Finish reason: %s\n", chunk.FinishReason)
			fmt.Printf("Prompt tokens: %d\n", chunk.PromptTokens)
			fmt.Printf("Completion tokens: %d\n", chunk.CompletionTokens)
			fmt.Printf("Cached tokens: %d\n", chunk.CachedTokens)

			// Print timing metrics
			fmt.Printf("\n--- Performance Metrics ---\n")
			fmt.Printf("TTFT (Time To First Token): %v\n", ttft)
			fmt.Printf("Total generation time: %v\n", totalDuration)
			if chunk.CompletionTokens > 0 && totalDuration > 0 {
				tokensPerSecond := float64(chunk.CompletionTokens) / totalDuration.Seconds()
				fmt.Printf("Throughput: %.2f tokens/second\n", tokensPerSecond)
			}

			// Print accumulated text
			fullText := accumulatedText.String()
			if fullText != "" {
				fmt.Printf("\n--- Full Response Text ---\n")
				fmt.Printf("%s\n", fullText)
			}

			// Print all tool calls found
			if len(allToolCalls) > 0 {
				fmt.Printf("\n--- All Tool Calls (Final) ---\n")
				for i, toolCall := range allToolCalls {
					fmt.Printf("\nTool Call #%d:\n", i+1)
					fmt.Printf("  ID: %s\n", toolCall.ID)
					fmt.Printf("  Type: %s\n", toolCall.Type)
					fmt.Printf("  Function: %s\n", toolCall.Function.Name)
					if toolCall.Function.Arguments != "" {
						fmt.Printf("  Arguments: %s\n", toolCall.Function.Arguments)

						// Try to pretty-print arguments if it's JSON
						var argsJSON interface{}
						if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &argsJSON); err == nil {
							prettyArgs, _ := json.MarshalIndent(argsJSON, "    ", "  ")
							fmt.Printf("  Arguments (formatted):\n%s\n", string(prettyArgs))
						}
					}
				}
			} else {
				fmt.Println("\n--- No Tool Calls Detected ---")
				fmt.Println("Note: The response may not contain tool calls, or the parser may not have detected them.")
				fmt.Println("Full response text is shown above for manual inspection.")
			}

			break
		}
	}
}

func showHelp() {
	fmt.Fprintf(os.Stderr, `Streaming Tool Call Example - Incremental Tool Call Parsing

Usage:
  %s -tokenizer <path> -endpoint <endpoint> [flags]

Required Flags:
  -tokenizer string
        Path to tokenizer directory or tokenizer.json file
        Recommended: specify directory (e.g., /path/to/tokenizer)

  -endpoint string
        gRPC endpoint address, e.g., grpc://localhost:20000

Optional Flags:
  -parser string
        Tool parser type (default: "auto")
        Options: auto, json, llama, mistral, step3, qwen, deepseek, kimik2, gpt_oss, pythonic

  -help
        Show this help message

Examples:
  # Using directory path (recommended)
  %s -tokenizer /path/to/tokenizer -endpoint grpc://localhost:20000

  # With specific parser type
  %s -tokenizer /path/to/tokenizer -endpoint grpc://localhost:20000 -parser json

Description:
  This example demonstrates streaming generation with incremental tool call parsing.
  It uses Rust FFI tool parser to parse tool calls as they arrive in the stream,
  allowing you to see tool calls being built incrementally.

Features:
  - Real-time streaming of text chunks
  - Incremental tool call parsing using Rust FFI
  - Tool call detection and argument accumulation
  - Performance metrics (TTFT, throughput)
  - Full response text and final tool calls

`, os.Args[0], os.Args[0], os.Args[0])
}
