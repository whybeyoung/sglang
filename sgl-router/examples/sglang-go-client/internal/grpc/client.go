// Package grpc provides gRPC client for SGLang scheduler
package grpc

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/sglang/sgl-router-go-client/proto"
)

// Client wraps the gRPC client for SGLang scheduler
type Client struct {
	conn   *grpc.ClientConn
	client proto.SglangSchedulerClient
}

// NewClient creates a new gRPC client
func NewClient(endpoint string, opts ...grpc.DialOption) (*Client, error) {
	// Convert grpc:// to plain address for gRPC
	addr := endpoint
	if len(addr) >= 7 && addr[:7] == "grpc://" {
		addr = addr[7:]
	}

	// Default options
	defaultOpts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
		grpc.WithTimeout(10 * time.Second),
	}

	// Merge with user-provided options
	allOpts := append(defaultOpts, opts...)

	// Create connection
	conn, err := grpc.NewClient(addr, allOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to connect: %w", err)
	}

	client := proto.NewSglangSchedulerClient(conn)

	return &Client{
		conn:   conn,
		client: client,
	}, nil
}

// Close closes the gRPC connection
func (c *Client) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// Generate sends a generation request and returns a stream
func (c *Client) Generate(ctx context.Context, req *proto.GenerateRequest) (grpc.ServerStreamingClient[proto.GenerateResponse], error) {
	return c.client.Generate(ctx, req)
}

// HealthCheck performs a health check
func (c *Client) HealthCheck(ctx context.Context) (*proto.HealthCheckResponse, error) {
	req := &proto.HealthCheckRequest{}
	return c.client.HealthCheck(ctx, req)
}

// GetModelInfo gets model information
func (c *Client) GetModelInfo(ctx context.Context) (*proto.GetModelInfoResponse, error) {
	req := &proto.GetModelInfoRequest{}
	return c.client.GetModelInfo(ctx, req)
}

// AbortRequest aborts a running request
func (c *Client) AbortRequest(ctx context.Context, requestID, reason string) error {
	req := &proto.AbortRequest{
		RequestId: requestID,
		Reason:    reason,
	}
	_, err := c.client.Abort(ctx, req)
	return err
}

// BuildGenerateRequest builds a GenerateRequest from parameters
func BuildGenerateRequest(
	requestID string,
	tokenIds []uint32,
	processedText string,
	options *GenerationOptions,
) (*proto.GenerateRequest, error) {
	// Build tokenized input
	tokenizedInput := &proto.TokenizedInput{
		OriginalText: processedText,
		InputIds:     tokenIds,
	}

	// Build sampling params
	samplingParams := &proto.SamplingParams{
		Temperature:                options.Temperature,
		TopP:                       options.TopP,
		TopK:                       options.TopK,
		MaxNewTokens:               options.MaxNewTokens,
		Stop:                       options.Stop,
		StopTokenIds:               options.StopTokenIds,
		SkipSpecialTokens:          options.SkipSpecialTokens,
		SpacesBetweenSpecialTokens: true,
		RepetitionPenalty:          1.0,
		N:                          1,
	}

	// Add constraint if provided
	if options.ConstraintType != "" && options.ConstraintValue != "" {
		switch options.ConstraintType {
		case "json_schema":
			samplingParams.Constraint = &proto.SamplingParams_JsonSchema{
				JsonSchema: options.ConstraintValue,
			}
		case "regex":
			samplingParams.Constraint = &proto.SamplingParams_Regex{
				Regex: options.ConstraintValue,
			}
		case "ebnf_grammar":
			samplingParams.Constraint = &proto.SamplingParams_EbnfGrammar{
				EbnfGrammar: options.ConstraintValue,
			}
		case "structural_tag":
			samplingParams.Constraint = &proto.SamplingParams_StructuralTag{
				StructuralTag: options.ConstraintValue,
			}
		}
	}

	// Build request
	req := &proto.GenerateRequest{
		RequestId:      requestID,
		Tokenized:      tokenizedInput,
		SamplingParams: samplingParams,
		ReturnLogprob:  options.ReturnLogprobs,
		TopLogprobsNum: int32(options.TopLogprobs),
		Stream:         options.Stream,
		LogMetrics:     false,
		Timestamp:      timestamppb.Now(),
	}

	return req, nil
}

// GenerationOptions contains options for generation
type GenerationOptions struct {
	Temperature       float32
	TopP              float32
	TopK              int32
	MaxNewTokens      *int32
	Stop              []string
	StopTokenIds      []uint32
	SkipSpecialTokens bool
	ReturnLogprobs    bool
	TopLogprobs       int
	Stream            bool
	ConstraintType    string // "json_schema", "regex", "ebnf_grammar", "structural_tag"
	ConstraintValue   string
}

// DefaultGenerationOptions returns default generation options
func DefaultGenerationOptions() *GenerationOptions {
	maxTokens := int32(100)
	return &GenerationOptions{
		Temperature:       1.0,
		TopP:              1.0,
		TopK:              -1,
		MaxNewTokens:      &maxTokens,
		Stop:              []string{},
		StopTokenIds:      []uint32{},
		SkipSpecialTokens: true,
		ReturnLogprobs:    false,
		TopLogprobs:       0,
		Stream:            false,
	}
}

// ProcessStreamResponse processes a streaming response
func ProcessStreamResponse(stream grpc.ServerStreamingClient[proto.GenerateResponse]) (*GenerateResult, error) {
	var allTokens []uint32
	var complete *proto.GenerateComplete
	var promptTokens int32
	var completionTokens int32
	var cachedTokens int32
	var outputLogprobs *proto.OutputLogProbs
	var inputLogprobs *proto.InputLogProbs

	// Track timing for TTFT and throughput
	startTime := time.Now()
	var firstTokenTime time.Time
	firstTokenReceived := false

	for {
		resp, err := stream.Recv()
		if err != nil {
			return nil, fmt.Errorf("stream receive error: %w", err)
		}

		// Handle different response types
		if chunk := resp.GetChunk(); chunk != nil {
			// Record time of first token
			if !firstTokenReceived && len(chunk.TokenIds) > 0 {
				firstTokenTime = time.Now()
				firstTokenReceived = true
			}
			// Accumulate tokens from chunk
			allTokens = append(allTokens, chunk.TokenIds...)
			promptTokens = chunk.PromptTokens
			completionTokens = chunk.CompletionTokens
			cachedTokens = chunk.CachedTokens
			if chunk.OutputLogprobs != nil {
				outputLogprobs = chunk.OutputLogprobs
			}
			if chunk.InputLogprobs != nil {
				inputLogprobs = chunk.InputLogprobs
			}
		} else if completeResp := resp.GetComplete(); completeResp != nil {
			// Record time of first token if not already recorded
			if !firstTokenReceived && len(completeResp.OutputIds) > 0 {
				firstTokenTime = time.Now()
				firstTokenReceived = true
			}
			// Final response
			complete = completeResp
			allTokens = completeResp.OutputIds
			promptTokens = completeResp.PromptTokens
			completionTokens = completeResp.CompletionTokens
			cachedTokens = completeResp.CachedTokens
			if completeResp.OutputLogprobs != nil {
				outputLogprobs = completeResp.OutputLogprobs
			}
			if completeResp.InputLogprobs != nil {
				inputLogprobs = completeResp.InputLogprobs
			}
			break
		} else if errResp := resp.GetError(); errResp != nil {
			return nil, fmt.Errorf("server error: %s (status: %s, details: %s)", errResp.GetMessage(), errResp.GetHttpStatusCode(), errResp.GetDetails())
		}
	}

	if complete == nil {
		return nil, fmt.Errorf("stream ended without complete response")
	}

	endTime := time.Now()
	totalDuration := endTime.Sub(startTime)
	var ttft time.Duration
	if firstTokenReceived {
		ttft = firstTokenTime.Sub(startTime)
	} else {
		// If no tokens received, TTFT equals total duration
		ttft = totalDuration
	}

	return &GenerateResult{
		TokenIds:         allTokens,
		FinishReason:     complete.FinishReason,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		CachedTokens:     cachedTokens,
		OutputLogprobs:   outputLogprobs,
		InputLogprobs:    inputLogprobs,
		TTFT:             ttft,
		TotalDuration:    totalDuration,
	}, nil
}

// GenerateResult contains the result of a generation request
type GenerateResult struct {
	TokenIds         []uint32
	FinishReason     string
	PromptTokens     int32
	CompletionTokens int32
	CachedTokens     int32
	OutputLogprobs   *proto.OutputLogProbs
	InputLogprobs    *proto.InputLogProbs
	TTFT             time.Duration // Time To First Token
	TotalDuration    time.Duration // Total generation time
}
