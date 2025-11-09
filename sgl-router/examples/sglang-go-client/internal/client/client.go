// Package client provides a high-level client for SGLang
package client

import (
	"context"
	"fmt"
	"time"

	"github.com/sglang/sgl-router-go-client/internal/grpc"
	"github.com/sglang/sgl-router-go-client/internal/tokenizer"
	"github.com/sglang/sgl-router-go-client/proto"
)

// Client is the main client for interacting with SGLang
type Client struct {
	tokenizer *tokenizer.Tokenizer
	grpc      *grpc.Client
}

// Config contains client configuration
type Config struct {
	TokenizerPath string
	GRPCEndpoint  string
	Timeout       time.Duration
}

// NewClient creates a new SGLang client
func NewClient(config Config) (*Client, error) {
	// Create tokenizer
	tok, err := tokenizer.NewTokenizer(config.TokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create tokenizer: %w", err)
	}

	// Create gRPC client
	grpcClient, err := grpc.NewClient(config.GRPCEndpoint)
	if err != nil {
		tok.Close()
		return nil, fmt.Errorf("failed to create gRPC client: %w", err)
	}

	return &Client{
		tokenizer: tok,
		grpc:      grpcClient,
	}, nil
}

// Close closes the client and releases resources
func (c *Client) Close() error {
	var errs []error

	if c.tokenizer != nil {
		if err := c.tokenizer.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	if c.grpc != nil {
		if err := c.grpc.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors closing client: %v", errs)
	}

	return nil
}

// Generate generates text from messages
func (c *Client) Generate(ctx context.Context, messages []tokenizer.Message, options *grpc.GenerationOptions) (*grpc.GenerateResult, error) {
	// Process messages
	tokenIds, processedText, err := c.tokenizer.ProcessMessages(messages)
	if err != nil {
		return nil, fmt.Errorf("failed to process messages: %w", err)
	}

	// Build request
	requestID := fmt.Sprintf("req-%d", time.Now().UnixNano())
	req, err := grpc.BuildGenerateRequest(requestID, tokenIds, processedText, options)
	if err != nil {
		return nil, fmt.Errorf("failed to build request: %w", err)
	}

	// Send request
	stream, err := c.grpc.Generate(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to start generation: %w", err)
	}

	// Process response
	result, err := grpc.ProcessStreamResponse(stream)
	if err != nil {
		return nil, fmt.Errorf("failed to process response: %w", err)
	}

	return result, nil
}

// GenerateText generates text from a simple text prompt
func (c *Client) GenerateText(ctx context.Context, prompt string, options *grpc.GenerationOptions) (*grpc.GenerateResult, error) {
	messages := []tokenizer.Message{
		{Role: "user", Content: prompt},
	}
	return c.Generate(ctx, messages, options)
}

// HealthCheck performs a health check
func (c *Client) HealthCheck(ctx context.Context) error {
	_, err := c.grpc.HealthCheck(ctx)
	return err
}

// GetModelInfo gets model information
func (c *Client) GetModelInfo(ctx context.Context) (*proto.GetModelInfoResponse, error) {
	return c.grpc.GetModelInfo(ctx)
}

// GetGRPCClient returns the underlying gRPC client for advanced usage
func (c *Client) GetGRPCClient() *grpc.Client {
	return c.grpc
}

// GetTokenizer returns the underlying tokenizer for advanced usage
func (c *Client) GetTokenizer() *tokenizer.Tokenizer {
	return c.tokenizer
}

// StreamChunk represents a single chunk in a streaming response
type StreamChunk struct {
	TokenIds         []uint32
	Text             string // Decoded text for this chunk
	IsComplete       bool   // True if this is the final chunk
	FinishReason     string
	PromptTokens     int32
	CompletionTokens int32
	CachedTokens     int32
	Error            error // Non-nil if there was an error
}

// GenerateStream generates text from messages in streaming mode
// Returns a channel that will receive stream chunks
func (c *Client) GenerateStream(ctx context.Context, messages []tokenizer.Message, options *grpc.GenerationOptions) (<-chan StreamChunk, error) {
	// Ensure streaming is enabled
	options.Stream = true

	// Process messages
	tokenIds, processedText, err := c.tokenizer.ProcessMessages(messages)
	if err != nil {
		return nil, fmt.Errorf("failed to process messages: %w", err)
	}

	// Build request
	requestID := fmt.Sprintf("req-%d", time.Now().UnixNano())
	req, err := grpc.BuildGenerateRequest(requestID, tokenIds, processedText, options)
	if err != nil {
		return nil, fmt.Errorf("failed to build request: %w", err)
	}

	// Send request
	stream, err := c.grpc.Generate(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to start generation: %w", err)
	}

	// Create channel for streaming chunks
	chunkChan := make(chan StreamChunk, 10)

	// Start goroutine to process stream
	go func() {
		defer close(chunkChan)

		for {
			resp, err := stream.Recv()
			if err != nil {
				// Send error chunk
				chunkChan <- StreamChunk{
					Error: fmt.Errorf("stream receive error: %w", err),
				}
				return
			}

			// Handle different response types
			if chunk := resp.GetChunk(); chunk != nil {
				// Decode tokens to text
				var decodedText string
				if len(chunk.TokenIds) > 0 {
					decoded, err := c.tokenizer.DecodeText(chunk.TokenIds, options.SkipSpecialTokens)
					if err != nil {
						// Try without skipping special tokens
						decoded, err2 := c.tokenizer.DecodeText(chunk.TokenIds, false)
						if err2 == nil {
							decodedText = decoded
						}
					} else {
						decodedText = decoded
					}
				}

				chunkChan <- StreamChunk{
					TokenIds:         chunk.TokenIds,
					Text:             decodedText,
					IsComplete:       false,
					PromptTokens:     chunk.PromptTokens,
					CompletionTokens: chunk.CompletionTokens,
					CachedTokens:     chunk.CachedTokens,
				}
			} else if completeResp := resp.GetComplete(); completeResp != nil {
				// Final response
				var decodedText string
				if len(completeResp.OutputIds) > 0 {
					decoded, err := c.tokenizer.DecodeText(completeResp.OutputIds, options.SkipSpecialTokens)
					if err != nil {
						// Try without skipping special tokens
						decoded, err2 := c.tokenizer.DecodeText(completeResp.OutputIds, false)
						if err2 == nil {
							decodedText = decoded
						}
					} else {
						decodedText = decoded
					}
				}

				chunkChan <- StreamChunk{
					TokenIds:         completeResp.OutputIds,
					Text:             decodedText,
					IsComplete:       true,
					FinishReason:     completeResp.FinishReason,
					PromptTokens:     completeResp.PromptTokens,
					CompletionTokens: completeResp.CompletionTokens,
					CachedTokens:     completeResp.CachedTokens,
				}
				return
			} else if errResp := resp.GetError(); errResp != nil {
				chunkChan <- StreamChunk{
					Error: fmt.Errorf("server error: %s (status: %s, details: %s)", errResp.GetMessage(), errResp.GetHttpStatusCode(), errResp.GetDetails()),
				}
				return
			}
		}
	}()

	return chunkChan, nil
}

// GenerateStreamWithTools generates text from messages with tools in streaming mode
func (c *Client) GenerateStreamWithTools(ctx context.Context, messages []tokenizer.Message, tools []tokenizer.Tool, options *grpc.GenerationOptions) (<-chan StreamChunk, error) {
	// Ensure streaming is enabled
	options.Stream = true

	// Process messages with tools
	tokenIds, processedText, err := c.tokenizer.ProcessMessagesWithTools(messages, tools)
	if err != nil {
		return nil, fmt.Errorf("failed to process messages: %w", err)
	}

	// Build request
	requestID := fmt.Sprintf("req-%d", time.Now().UnixNano())
	req, err := grpc.BuildGenerateRequest(requestID, tokenIds, processedText, options)
	if err != nil {
		return nil, fmt.Errorf("failed to build request: %w", err)
	}

	// Send request
	stream, err := c.grpc.Generate(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to start generation: %w", err)
	}

	// Create channel for streaming chunks
	chunkChan := make(chan StreamChunk, 10)

	// Start goroutine to process stream
	go func() {
		defer close(chunkChan)

		for {
			resp, err := stream.Recv()
			if err != nil {
				// Send error chunk
				chunkChan <- StreamChunk{
					Error: fmt.Errorf("stream receive error: %w", err),
				}
				return
			}

			// Handle different response types
			if chunk := resp.GetChunk(); chunk != nil {
				// Decode tokens to text
				var decodedText string
				if len(chunk.TokenIds) > 0 {
					// For tool calls, don't skip special tokens
					decoded, err := c.tokenizer.DecodeText(chunk.TokenIds, false)
					if err == nil {
						decodedText = decoded
					}
				}

				chunkChan <- StreamChunk{
					TokenIds:         chunk.TokenIds,
					Text:             decodedText,
					IsComplete:       false,
					PromptTokens:     chunk.PromptTokens,
					CompletionTokens: chunk.CompletionTokens,
					CachedTokens:     chunk.CachedTokens,
				}
			} else if completeResp := resp.GetComplete(); completeResp != nil {
				// Final response
				var decodedText string
				if len(completeResp.OutputIds) > 0 {
					// For tool calls, don't skip special tokens
					decoded, err := c.tokenizer.DecodeText(completeResp.OutputIds, false)
					if err == nil {
						decodedText = decoded
					}
				}

				chunkChan <- StreamChunk{
					TokenIds:         completeResp.OutputIds,
					Text:             decodedText,
					IsComplete:       true,
					FinishReason:     completeResp.FinishReason,
					PromptTokens:     completeResp.PromptTokens,
					CompletionTokens: completeResp.CompletionTokens,
					CachedTokens:     completeResp.CachedTokens,
				}
				return
			} else if errResp := resp.GetError(); errResp != nil {
				chunkChan <- StreamChunk{
					Error: fmt.Errorf("server error: %s (status: %s, details: %s)", errResp.GetMessage(), errResp.GetHttpStatusCode(), errResp.GetDetails()),
				}
				return
			}
		}
	}()

	return chunkChan, nil
}
