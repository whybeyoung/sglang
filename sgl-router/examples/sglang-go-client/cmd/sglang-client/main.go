// Package main provides an example usage of the SGLang Go client
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/sglang/sgl-router-go-client/config"
	"github.com/sglang/sgl-router-go-client/internal/client"
	"github.com/sglang/sgl-router-go-client/internal/grpc"
	"github.com/sglang/sgl-router-go-client/internal/tokenizer"
)

func main() {
	// Parse command line flags
	var (
		tokenizerPath = flag.String("tokenizer", "", "Path to tokenizer directory or tokenizer.json file (required). "+
			"Recommended: specify directory (e.g., /path/to/tokenizer) to auto-discover tokenizer.json and tokenizer_config.json")
		grpcEndpoint = flag.String("endpoint", "", "gRPC endpoint address, e.g., grpc://localhost:20000 (required)")
		configFile   = flag.String("config", "", "Path to config JSON file (optional)")
		help         = flag.Bool("help", false, "Show help message")
	)
	flag.Parse()

	// Show help if requested
	if *help {
		showHelp()
		os.Exit(0)
	}

	// Load configuration
	var cfg *config.Config
	var err error

	if *configFile != "" {
		// Load from config file
		cfg, err = config.LoadFromFile(*configFile)
		if err != nil {
			log.Fatalf("Failed to load config file: %v", err)
		}
	} else {
		// Use default config
		cfg = config.DefaultConfig()
	}

	// Override with environment variables if available
	if envCfg := config.LoadFromEnv(); envCfg.Tokenizer.Path != "" {
		cfg.Tokenizer.Path = envCfg.Tokenizer.Path
	}
	if envCfg := config.LoadFromEnv(); envCfg.GRPC.Endpoint != "" {
		cfg.GRPC.Endpoint = envCfg.GRPC.Endpoint
	}

	// Override with command line arguments (highest priority)
	if *tokenizerPath != "" {
		cfg.Tokenizer.Path = *tokenizerPath
	}
	if *grpcEndpoint != "" {
		cfg.GRPC.Endpoint = *grpcEndpoint
	}

	// Validate required parameters
	if cfg.Tokenizer.Path == "" || cfg.Tokenizer.Path == "/path/to/tokenizer.json" {
		log.Fatal("Error: tokenizer path is required. Use -tokenizer flag or set SGL_TOKENIZER_PATH environment variable.\n" +
			"Recommended: specify directory path (e.g., /path/to/tokenizer) to auto-discover tokenizer files")
	}
	if cfg.GRPC.Endpoint == "" || cfg.GRPC.Endpoint == "grpc://localhost:20000" {
		log.Fatal("Error: gRPC endpoint is required. Use -endpoint flag or set SGL_GRPC_ENDPOINT environment variable")
	}

	// Create client
	clientConfig := client.Config{
		TokenizerPath: cfg.Tokenizer.Path,
		GRPCEndpoint:  cfg.GRPC.Endpoint,
		Timeout:       time.Duration(cfg.GRPC.Timeout) * time.Second,
	}

	cl, err := client.NewClient(clientConfig)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer cl.Close()

	// Health check
	fmt.Println("Performing health check...")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := cl.HealthCheck(ctx); err != nil {
		log.Printf("Health check failed: %v", err)
	} else {
		fmt.Println("✓ Health check passed")
	}

	// Prepare messages
	messages := []tokenizer.Message{
		{
			Role:    "system",
			Content: "You are a helpful assistant.",
		},
		{
			Role:    "user",
			Content: "What is the capital of France?",
		},
	}

	// Set generation options
	options := grpc.DefaultGenerationOptions()
	options.Temperature = cfg.Generation.Temperature
	options.TopP = cfg.Generation.TopP
	options.TopK = cfg.Generation.TopK
	maxTokens := cfg.Generation.MaxNewTokens
	options.MaxNewTokens = &maxTokens
	options.SkipSpecialTokens = cfg.Generation.SkipSpecialTokens
	options.Stream = cfg.Generation.Stream

	// Generate
	fmt.Println("\nGenerating response...")
	genCtx, genCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer genCancel()

	result, err := cl.Generate(genCtx, messages, options)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	// Print results
	fmt.Printf("\n✓ Generation complete!\n")
	fmt.Printf("Finish reason: %s\n", result.FinishReason)
	fmt.Printf("Prompt tokens: %d\n", result.PromptTokens)
	fmt.Printf("Completion tokens: %d\n", result.CompletionTokens)
	fmt.Printf("Cached tokens: %d\n", result.CachedTokens)

	// Print timing metrics
	fmt.Printf("\n--- Performance Metrics ---\n")
	fmt.Printf("TTFT (Time To First Token): %v\n", result.TTFT)
	fmt.Printf("Total generation time: %v\n", result.TotalDuration)
	if result.CompletionTokens > 0 && result.TotalDuration > 0 {
		// Calculate tokens per second (throughput)
		tokensPerSecond := float64(result.CompletionTokens) / result.TotalDuration.Seconds()
		fmt.Printf("Throughput: %.2f tokens/second\n", tokensPerSecond)
		// Calculate time per token
		timePerToken := result.TotalDuration / time.Duration(result.CompletionTokens)
		fmt.Printf("Time per token: %v\n", timePerToken)
	}

	// Decode and print the response text
	if len(result.TokenIds) > 0 {
		// Get tokenizer from client to decode
		// Note: We need to access the tokenizer, but it's private in the client
		// For now, we'll create a temporary tokenizer for decoding
		tok, err := tokenizer.NewTokenizer(cfg.Tokenizer.Path)
		if err != nil {
			log.Printf("Warning: Failed to create tokenizer for decoding: %v", err)
			fmt.Printf("Generated token IDs (%d): %v\n", len(result.TokenIds), result.TokenIds[:min(20, len(result.TokenIds))])
		} else {
			defer tok.Close()
			// Decode with and without special tokens to see both versions
			decodedText, err := tok.DecodeText(result.TokenIds, true)
			if err != nil {
				log.Printf("Warning: Failed to decode tokens: %v", err)
				fmt.Printf("Generated token IDs (%d): %v\n", len(result.TokenIds), result.TokenIds[:min(20, len(result.TokenIds))])
			} else {
				if decodedText != "" {
					fmt.Printf("\nGenerated response:\n%s\n", decodedText)
				} else {
					// If decoded text is empty, try without skipping special tokens
					decodedWithSpecial, err2 := tok.DecodeText(result.TokenIds, false)
					if err2 == nil && decodedWithSpecial != "" {
						fmt.Printf("\nGenerated response (with special tokens):\n%s\n", decodedWithSpecial)
					} else {
						fmt.Printf("\nGenerated token IDs (%d): %v\n", len(result.TokenIds), result.TokenIds)
						if len(result.TokenIds) <= 20 {
							fmt.Printf("(Response may be empty or contain only special tokens)\n")
						}
					}
				}
			}
		}
	} else {
		fmt.Println("No tokens generated")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func showHelp() {
	fmt.Fprintf(os.Stderr, `SGLang Go Client Example

Usage:
  %s [flags]

Flags:
  -tokenizer string
        Path to tokenizer directory or tokenizer.json file (required)
        Recommended: specify directory (e.g., /path/to/tokenizer) to auto-discover:
          - tokenizer.json (required)
          - tokenizer_config.json (for chat template, optional)
        Can also be set via SGL_TOKENIZER_PATH environment variable

  -endpoint string
        gRPC endpoint address, e.g., grpc://localhost:20000 (required)
        Can also be set via SGL_GRPC_ENDPOINT environment variable

  -config string
        Path to config JSON file (optional)
        Command line flags override config file values

  -help
        Show this help message

Examples:
  # Using directory path (recommended - auto-discovers tokenizer.json and tokenizer_config.json)
  %s -tokenizer /path/to/tokenizer -endpoint grpc://localhost:20000

  # Using file path (also works)
  %s -tokenizer /path/to/tokenizer.json -endpoint grpc://localhost:20000

  # Using environment variables
  export SGL_TOKENIZER_PATH=/path/to/tokenizer
  export SGL_GRPC_ENDPOINT=grpc://localhost:20000
  %s

  # Using config file
  %s -config config.json

  # Mix: config file with override
  %s -config config.json -endpoint grpc://localhost:30000

`, os.Args[0], os.Args[0], os.Args[0], os.Args[0], os.Args[0])
}
