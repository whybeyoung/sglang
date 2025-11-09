// Simple example demonstrating basic usage
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
)

func main() {
	// Parse command line flags
	var (
		tokenizerPath = flag.String("tokenizer", "", "Path to tokenizer directory or tokenizer.json file (required). "+
			"Recommended: specify directory to auto-discover tokenizer files")
		grpcEndpoint = flag.String("endpoint", "", "gRPC endpoint address, e.g., grpc://localhost:20000 (required)")
		prompt       = flag.String("prompt", "Hello, how are you?", "Text prompt to generate")
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

	// Simple text generation
	ctx := context.Background()
	result, err := cl.GenerateText(ctx, *prompt, grpc.DefaultGenerationOptions())
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	fmt.Printf("Generated %d tokens\n", len(result.TokenIds))
	fmt.Printf("Finish reason: %s\n", result.FinishReason)
}

func showHelp() {
	fmt.Fprintf(os.Stderr, `Simple SGLang Client Example

Usage:
  %s -tokenizer <path> -endpoint <endpoint> [flags]

Required Flags:
  -tokenizer string
        Path to tokenizer directory or tokenizer.json file
        Recommended: specify directory (e.g., /path/to/tokenizer)

  -endpoint string
        gRPC endpoint address, e.g., grpc://localhost:20000

Optional Flags:
  -prompt string
        Text prompt to generate (default: "Hello, how are you?")

  -help
        Show this help message

Examples:
  # Using directory path (recommended)
  %s -tokenizer /path/to/tokenizer -endpoint grpc://localhost:20000
  
  # Using file path (also works)
  %s -tokenizer /path/to/tokenizer.json -endpoint grpc://localhost:20000
  
  # With custom prompt
  %s -tokenizer /path/to/tokenizer -endpoint grpc://localhost:20000 -prompt "What is AI?"

`, os.Args[0], os.Args[0], os.Args[0])
}
