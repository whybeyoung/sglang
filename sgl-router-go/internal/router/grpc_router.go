package router

import (
	"context"
	"fmt"

	"github.com/sglang/sglang-router-go/internal/core"
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
	logger           *zap.Logger
}

// NewGrpcRouter creates a new gRPC router
func NewGrpcRouter(
	workerRegistry *core.WorkerRegistry,
	policyRegistry interface{}, // TODO: Define PolicyRegistry type
	tokenizer interface{}, // TODO: Define Tokenizer type
	toolParserFactory interface{}, // TODO: Define factory type
	reasoningParserFactory interface{}, // TODO: Define factory type
	logger *zap.Logger,
) (*GrpcRouter, error) {
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
		stages.NewClientAcquisitionStage(logger),
		stages.NewRequestBuildingStage(false, logger), // No PD metadata
		stages.NewDispatchMetadataStage(logger),
		// TODO: Add RequestExecutionStage
		// TODO: Add ResponseProcessingStage
	}

	requestPipeline := pipeline.NewPipeline(pipelineStages, logger)

	return &GrpcRouter{
		workerRegistry:   workerRegistry,
		requestPipeline:  requestPipeline,
		sharedComponents: sharedComponents,
		logger:           logger,
	}, nil
}

// RouteChat routes a chat completion request
// Similar to Rust route_chat_impl
func (r *GrpcRouter) RouteChat(
	ctx context.Context,
	request interface{}, // TODO: Define ChatCompletionRequest type
	modelID *string,
) (interface{}, error) {
	r.logger.Debug("Processing chat completion request",
		zap.Stringp("model_id", modelID),
	)

	// Create request input
	input := &pipeline.RequestInput{
		RequestType: pipeline.RequestTypeChat,
		ModelID:     modelID,
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

	return response, nil
}

// RouteGenerate routes a generate request
// Similar to Rust route_generate_impl
func (r *GrpcRouter) RouteGenerate(
	ctx context.Context,
	request interface{}, // TODO: Define GenerateRequest type
	modelID *string,
) (interface{}, error) {
	r.logger.Debug("Processing generate request",
		zap.Stringp("model_id", modelID),
	)

	// Create request input
	input := &pipeline.RequestInput{
		RequestType: pipeline.RequestTypeGenerate,
		ModelID:     modelID,
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

	return response, nil
}
