package pipeline

import (
	"context"

	"go.uber.org/zap"
)

// PipelineStage represents a stage in the request processing pipeline
// Similar to Rust PipelineStage trait
type PipelineStage interface {
	// Execute executes this stage, mutating the context
	// Returns:
	//   - (nil, nil) - Continue to next stage
	//   - (response, nil) - Pipeline complete, return this response (e.g., streaming)
	//   - (nil, error) - Error occurred, return this error
	Execute(ctx *RequestContext) (interface{}, error)

	// Name returns the stage name for logging
	Name() string
}

// BaseStage provides common functionality for stages
type BaseStage struct {
	name   string
	Logger *zap.Logger // Exported so stages can access it
}

func NewBaseStage(name string, logger *zap.Logger) *BaseStage {
	return &BaseStage{
		name:   name,
		Logger: logger,
	}
}

func (s *BaseStage) Name() string {
	return s.name
}

// Pipeline orchestrates the execution of pipeline stages
// Similar to Rust RequestPipeline
type Pipeline struct {
	stages []PipelineStage
	logger *zap.Logger
}

// NewPipeline creates a new pipeline with the given stages
func NewPipeline(stages []PipelineStage, logger *zap.Logger) *Pipeline {
	return &Pipeline{
		stages: stages,
		logger: logger,
	}
}

// Execute executes the complete pipeline for a request context
func (p *Pipeline) Execute(ctx *RequestContext) (interface{}, error) {
	for idx, stage := range p.stages {
		p.logger.Debug("Executing pipeline stage",
			zap.Int("stage_index", idx),
			zap.String("stage_name", stage.Name()),
		)

		response, err := stage.Execute(ctx)
		if err != nil {
			p.logger.Error("Pipeline stage failed",
				zap.Int("stage_index", idx),
				zap.String("stage_name", stage.Name()),
				zap.Error(err),
			)
			return nil, err
		}

		// If stage returned a response, pipeline is complete
		if response != nil {
			p.logger.Debug("Pipeline completed early with response",
				zap.String("stage_name", stage.Name()),
			)
			return response, nil
		}

		// Continue to next stage
	}

	// All stages completed, extract final response from context
	// TODO: Extract final response from ctx.State.Response.FinalResponse
	return nil, nil
}

// ExecuteWithContext creates a context and executes the pipeline
func (p *Pipeline) ExecuteWithContext(
	parentCtx context.Context,
	input *RequestInput,
	components *SharedComponents,
	logger *zap.Logger,
) (interface{}, error) {
	ctx := NewRequestContext(parentCtx, input, components, logger)
	return p.Execute(ctx)
}
