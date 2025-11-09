// Chat example demonstrating multi-turn conversation
package main

import (
	"context"
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

	// Multi-turn conversation
	ctx := context.Background()
	options := grpc.DefaultGenerationOptions()
	maxTokens := int32(150)
	options.MaxNewTokens = &maxTokens

	// First turn
	messages1 := []tokenizer.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What is 2+2?"},
	}

	result1, err := cl.Generate(ctx, messages1, options)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	fmt.Printf("Assistant: Generated %d tokens\n", len(result1.TokenIds))

	// Second turn (continuing conversation)
	messages2 := []tokenizer.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What is 2+2?"},
		{Role: "assistant", Content: "4"}, // Previous response
		{Role: "user", Content: "What about 3+3?"},
	}

	result2, err := cl.Generate(ctx, messages2, options)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	fmt.Printf("Assistant: Generated %d tokens\n", len(result2.TokenIds))
}

func showHelp() {
	fmt.Fprintf(os.Stderr, `Chat Example - Multi-turn Conversation

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
  
  # Using file path (also works)
  %s -tokenizer /path/to/tokenizer.json -endpoint grpc://localhost:20000

`, os.Args[0], os.Args[0])
}
