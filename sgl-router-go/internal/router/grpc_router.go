package router

import (
	"context"
	"fmt"

	"github.com/sglang/sglang-router-go/internal/core"
	"github.com/sglang/sglang-router-go/internal/grpc"
	"github.com/sglang/sglang-router-go/internal/policy"
	"github.com/sglang/sglang-router-go/internal/protocols"
	"github.com/sglang/sglang-router-go/internal/router/pipeline"
	"github.com/sglang/sglang-router-go/internal/router/pipeline/stages"
	"go.uber.org/zap"
)

// GrpcRouter is the main gRPC router implementation
// Similar to Rust GrpcRouter
type GrpcRouter struct {
	workerRegistry   *core.WorkerRegistry
	requestPipeline  *pipeline.Pipeline
	sharedComponents *pipeline.SharedComponents
	clientPool       *grpc.ClientPool
	logger           *zap.Logger
}

// NewGrpcRouter creates a new gRPC router
func NewGrpcRouter(
	workerRegistry *core.WorkerRegistry,
	policyRegistry *policy.PolicyRegistry,
	tokenizer interface{}, // TODO: Define Tokenizer type
	toolParserFactory interface{}, // TODO: Define factory type
	reasoningParserFactory interface{}, // TODO: Define factory type
	logger *zap.Logger,
) (*GrpcRouter, error) {
	// Create client pool
	clientPool := grpc.NewClientPool(logger)

	// Create shared components
	sharedComponents := &pipeline.SharedComponents{
		Tokenizer:              tokenizer,
		ToolParserFactory:      toolParserFactory,
		ReasoningParserFactory: reasoningParserFactory,
	}

	// Create regular pipeline (single-worker mode)
	// Similar to Rust RequestPipeline::new_regular
	pipelineStages := []pipeline.PipelineStage{
		stages.NewPreparationStage(logger),
		stages.NewWorkerSelectionStage(
			workerRegistry,
			policyRegistry,
			stages.WorkerSelectionModeRegular,
			logger,
		),
		stages.NewClientAcquisitionStage(clientPool, logger),
		stages.NewRequestBuildingStage(false, logger), // No PD metadata
		stages.NewDispatchMetadataStage(logger),
		stages.NewRequestExecutionStage(stages.ExecutionModeSingle, logger),
		stages.NewResponseProcessingStage(logger),
	}

	requestPipeline := pipeline.NewPipeline(pipelineStages, logger)

	return &GrpcRouter{
		workerRegistry:   workerRegistry,
		requestPipeline:  requestPipeline,
		sharedComponents: sharedComponents,
		clientPool:       clientPool,
		logger:           logger,
	}, nil
}

// RouteChat routes a chat completion request
// Similar to Rust route_chat_impl
// Returns either:
// - *protocols.ChatCompletionResponse for non-streaming
// - StreamingResponse (SSE) for streaming requests
func (r *GrpcRouter) RouteChat(
	ctx context.Context,
	request *protocols.ChatCompletionRequest,
	modelID *string,
) (interface{}, error) {
	r.logger.Debug("Processing chat completion request",
		zap.Stringp("model_id", modelID),
	)

	// Create request input
	input := &pipeline.RequestInput{
		RequestType: pipeline.RequestTypeChat,
		ModelID:     modelID,
		Request:     request, // Store request for parameter extraction
	}

	// Execute pipeline
	response, err := r.requestPipeline.ExecuteWithContext(
		ctx,
		input,
		r.sharedComponents,
		r.logger,
	)

	if err != nil {
		return nil, fmt.Errorf("pipeline execution failed: %w", err)
	}

	// Check if this is a streaming response
	// In Rust, streaming responses are returned directly as HTTP Response
	// In Go, we return a StreamingResponse wrapper for HTTP server to handle
	if streamResp, ok := response.(*pipeline.StreamingResponse); ok {
		// Convert pipeline.StreamingResponse to router.StreamingResponse
		return &StreamingResponse{
			Reader:      streamResp.Reader,
			ContentType: streamResp.ContentType,
			Headers:     streamResp.Headers,
		}, nil
	}

	// Non-streaming response
	chatResponse, ok := response.(*protocols.ChatCompletionResponse)
	if !ok {
		return nil, fmt.Errorf("unexpected response type: %T", response)
	}

	return chatResponse, nil
}

// RouteGenerate routes a generate request
// Similar to Rust route_generate_impl
// Returns either:
// - *protocols.GenerateResponse for non-streaming
// - StreamingResponse (SSE) for streaming requests
func (r *GrpcRouter) RouteGenerate(
	ctx context.Context,
	request *protocols.GenerateRequest,
	modelID *string,
) (interface{}, error) {
	r.logger.Debug("Processing generate request",
		zap.Stringp("model_id", modelID),
	)

	// Create request input
	input := &pipeline.RequestInput{
		RequestType: pipeline.RequestTypeGenerate,
		ModelID:     modelID,
		Request:     request, // Store request for parameter extraction
	}

	// Execute pipeline
	response, err := r.requestPipeline.ExecuteWithContext(
		ctx,
		input,
		r.sharedComponents,
		r.logger,
	)

	if err != nil {
		return nil, fmt.Errorf("pipeline execution failed: %w", err)
	}

	// Check if this is a streaming response
	if streamResp, ok := response.(*pipeline.StreamingResponse); ok {
		// Convert pipeline.StreamingResponse to router.StreamingResponse
		return &StreamingResponse{
			Reader:      streamResp.Reader,
			ContentType: streamResp.ContentType,
			Headers:     streamResp.Headers,
		}, nil
	}

	// Non-streaming response
	generateResponse, ok := response.(*protocols.GenerateResponse)
	if !ok {
		return nil, fmt.Errorf("unexpected response type: %T", response)
	}

	return generateResponse, nil
}
