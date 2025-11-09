// Streaming example demonstrating real-time text generation
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
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
			"Recommended: specify directory to auto-discover tokenizer files")
		grpcEndpoint = flag.String("endpoint", "", "gRPC endpoint address, e.g., grpc://localhost:20000 (required)")
		configFile   = flag.String("config", "", "Path to config JSON file (optional)")
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

	if *configFile != "" {
		cfg, err = config.LoadFromFile(*configFile)
		if err != nil {
			log.Fatalf("Failed to load config file: %v", err)
		}
	} else {
		cfg = config.DefaultConfig()
	}

	// Override with command line arguments
	if *tokenizerPath != "" {
		cfg.Tokenizer.Path = *tokenizerPath
	}
	if *grpcEndpoint != "" {
		cfg.GRPC.Endpoint = *grpcEndpoint
	}

	// Validate required parameters
	if cfg.Tokenizer.Path == "" {
		log.Fatal("Error: tokenizer path is required. Use -tokenizer flag or set SGL_TOKENIZER_PATH environment variable")
	}
	if cfg.GRPC.Endpoint == "" {
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

	// Prepare messages
	messages := []tokenizer.Message{
		{
			Role:    "system",
			Content: "You are a helpful assistant.",
		},
		{
			Role:    "user",
			Content: "Write a short story about a robot learning to paint.",
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
	options.Stream = true // Enable streaming

	// Start streaming generation
	fmt.Println("Starting streaming generation...")
	fmt.Println("---")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	chunkChan, err := cl.GenerateStream(ctx, messages, options)
	if err != nil {
		log.Fatalf("Failed to start streaming: %v", err)
	}

	// Track metrics
	startTime := time.Now()
	var firstTokenTime time.Time
	firstTokenReceived := false
	var totalTokens int
	var fullText strings.Builder

	// Process stream chunks
	for chunk := range chunkChan {
		if chunk.Error != nil {
			log.Fatalf("Stream error: %v", chunk.Error)
		}

		// Record first token time
		if !firstTokenReceived && len(chunk.TokenIds) > 0 {
			firstTokenTime = time.Now()
			firstTokenReceived = true
			ttft := firstTokenTime.Sub(startTime)
			fmt.Printf("✓ First token received (TTFT: %v)\n", ttft)
			fmt.Println("---")
		}

		// Print text as it arrives
		if chunk.Text != "" {
			fmt.Print(chunk.Text) // Print without newline for smooth streaming
			fullText.WriteString(chunk.Text)
			totalTokens += len(chunk.TokenIds)
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
				timePerToken := totalDuration / time.Duration(chunk.CompletionTokens)
				fmt.Printf("Time per token: %v\n", timePerToken)
			}

			break
		}
	}
}

func showHelp() {
	fmt.Fprintf(os.Stderr, `Streaming Example - Real-time Text Generation

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
  # Using directory path (recommended)
  %s -tokenizer /path/to/tokenizer -endpoint grpc://localhost:20000

  # Using environment variables
  export SGL_TOKENIZER_PATH=/path/to/tokenizer
  export SGL_GRPC_ENDPOINT=grpc://localhost:20000
  %s

  # Using config file
  %s -config config.json

Description:
  This example demonstrates real-time streaming text generation.
  Text is printed as it arrives from the server, providing a live
  streaming experience similar to ChatGPT.

Features:
  - Real-time text streaming
  - Performance metrics (TTFT, throughput)
  - Token usage statistics

`, os.Args[0], os.Args[0], os.Args[0], os.Args[0])
}
