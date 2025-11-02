package stages

import (
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/sglang/sglang-router-go/internal/router/pipeline"
	"go.uber.org/zap"
)

// DispatchMetadataStage prepares dispatch metadata
// Similar to Rust DispatchMetadataStage
type DispatchMetadataStage struct {
	*pipeline.BaseStage
}

// NewDispatchMetadataStage creates a new dispatch metadata stage
func NewDispatchMetadataStage(logger *zap.Logger) *DispatchMetadataStage {
	return &DispatchMetadataStage{
		BaseStage: pipeline.NewBaseStage("DispatchMetadata", logger),
	}
}

// Execute implements PipelineStage
func (s *DispatchMetadataStage) Execute(ctx *pipeline.RequestContext) (interface{}, error) {
	protoRequest := ctx.State.ProtoRequest
	if protoRequest == nil {
		return nil, fmt.Errorf("request building stage not completed")
	}

	// Generate request ID
	requestID := uuid.New().String()

	// Get model ID
	modelID := "unknown"
	if ctx.Input.ModelID != nil {
		modelID = *ctx.Input.ModelID
	}

	// Determine if streaming (from request type - TODO: get from actual request)
	isStreaming := false // TODO: Extract from request

	ctx.State.Dispatch = &pipeline.DispatchMetadata{
		RequestID:     requestID,
		Model:         modelID,
		Created:       time.Now().Unix(),
		WeightVersion: nil, // TODO: Extract from worker metadata
		IsStreaming:   isStreaming,
	}

	s.Logger.Debug("Dispatch metadata prepared",
		zap.String("request_id", requestID),
		zap.String("model", modelID),
		zap.Bool("is_streaming", isStreaming),
	)

	return nil, nil // Continue to next stage
}
