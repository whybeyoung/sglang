// Function call example demonstrating tool calling
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/sglang/sgl-router-go-client/internal/client"
	"github.com/sglang/sgl-router-go-client/internal/grpc"
	"github.com/sglang/sgl-router-go-client/internal/tokenizer"
)

func main() {
	// Parse command line flags
	var (
		tokenizerPath = flag.String("tokenizer", "", "Path to tokenizer directory or tokenizer.json file (required). "+
			"Recommended: specify directory to auto-discover tokenizer files")
		grpcEndpoint = flag.String("endpoint", "", "gRPC endpoint address, e.g., grpc://localhost:20000 (required)")
		help         = flag.Bool("help", false, "Show help message")
	)
	flag.Parse()

	if *help {
		showHelp()
		os.Exit(0)
	}

	// Validate required parameters
	if *tokenizerPath == "" {
		log.Fatal("Error: -tokenizer flag is required")
	}
	if *grpcEndpoint == "" {
		log.Fatal("Error: -endpoint flag is required")
	}

	// Create client
	cfg := client.Config{
		TokenizerPath: *tokenizerPath,
		GRPCEndpoint:  *grpcEndpoint,
		Timeout:       30 * time.Second,
	}

	cl, err := client.NewClient(cfg)
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

	// Process messages with tools
	tok := cl.GetTokenizer()
	tokenIds, processedText, err := tok.ProcessMessagesWithTools(messages, tools)
	if err != nil {
		log.Fatalf("Failed to process messages: %v", err)
	}

	fmt.Printf("Processed text (first 200 chars): %s...\n", processedText[:min(200, len(processedText))])
	fmt.Printf("Token IDs count: %d\n", len(tokenIds))

	// Build generation options
	options := grpc.DefaultGenerationOptions()
	maxTokens := int32(200)
	options.MaxNewTokens = &maxTokens
	options.SkipSpecialTokens = false // Important: don't skip special tokens for tool calls

	// Build request manually to include tools constraint
	// Note: In a full implementation, you would generate tool constraints from tools
	// For now, we'll just send the request and decode the response
	requestID := fmt.Sprintf("req-%d", time.Now().UnixNano())
	req, err := grpc.BuildGenerateRequest(requestID, tokenIds, processedText, options)
	if err != nil {
		log.Fatalf("Failed to build request: %v", err)
	}

	// Send request
	stream, err := cl.GetGRPCClient().Generate(ctx, req)
	if err != nil {
		log.Fatalf("Failed to start generation: %v", err)
	}

	// Process response
	result, err := grpc.ProcessStreamResponse(stream)
	if err != nil {
		log.Fatalf("Failed to process response: %v", err)
	}

	// Decode response
	responseText, err := tok.DecodeText(result.TokenIds, false)
	if err != nil {
		log.Fatalf("Failed to decode response: %v", err)
	}

	fmt.Println("\n=== Response ===")
	fmt.Println(responseText)

	// Try to parse tool calls from response
	// Tool calls are typically in XML format like <tool_call>...</tool_call>
	// or JSON format depending on the model
	fmt.Println("\n=== Parsing Tool Calls ===")
	parseToolCalls(responseText)

	// Print metrics
	if result.TTFT > 0 {
		fmt.Printf("\n=== Metrics ===\n")
		fmt.Printf("TTFT (Time To First Token): %v\n", result.TTFT)
		if result.TotalDuration > 0 && len(result.TokenIds) > 0 {
			tokensPerSec := float64(len(result.TokenIds)) / result.TotalDuration.Seconds()
			fmt.Printf("Tokens per second: %.2f\n", tokensPerSec)
			fmt.Printf("Total tokens: %d\n", len(result.TokenIds))
		}
	}
}

func parseToolCalls(text string) {
	// Simple parsing for tool calls
	// In practice, you would use a proper XML or JSON parser
	// depending on the model's output format

	// Look for XML-style tool calls
	if contains(text, "<tool_call>") {
		fmt.Println("Found XML-style tool calls")
		// Extract tool call content
		start := indexOf(text, "<tool_call>")
		end := indexOf(text, "</tool_call>")
		if start >= 0 && end > start {
			toolCallContent := text[start+11 : end] // +11 for "<tool_call>"
			fmt.Printf("Tool call content: %s\n", toolCallContent)
		}
	}

	// Look for JSON-style tool calls
	if contains(text, "\"tool_calls\"") || contains(text, "\"function\"") {
		fmt.Println("Found JSON-style tool calls")
		// Try to parse as JSON
		var toolCalls struct {
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		}
		if err := json.Unmarshal([]byte(text), &toolCalls); err == nil {
			for _, tc := range toolCalls.ToolCalls {
				fmt.Printf("Tool Call:\n")
				fmt.Printf("  ID: %s\n", tc.ID)
				fmt.Printf("  Type: %s\n", tc.Type)
				fmt.Printf("  Function: %s\n", tc.Function.Name)
				fmt.Printf("  Arguments: %s\n", tc.Function.Arguments)
			}
		}
	}
}

func contains(s, substr string) bool {
	return indexOf(s, substr) >= 0
}

func indexOf(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func showHelp() {
	fmt.Fprintf(os.Stderr, `Function Call Example - Tool Calling

Usage:
  %s -tokenizer <path> -endpoint <endpoint>

Required Flags:
  -tokenizer string
        Path to tokenizer directory or tokenizer.json file
        Recommended: specify directory (e.g., /path/to/tokenizer)

  -endpoint string
        gRPC endpoint address, e.g., grpc://localhost:20000

  -help
        Show this help message

Examples:
  # Using directory path (recommended)
  %s -tokenizer /path/to/tokenizer -endpoint grpc://localhost:20000

Description:
  This example demonstrates how to use function calling (tool calling) with SGLang.
  It defines two tools (get_weather and get_time) and sends a request that should
  trigger a tool call. The response is decoded and parsed to extract tool call information.

`, os.Args[0], os.Args[0])
}
