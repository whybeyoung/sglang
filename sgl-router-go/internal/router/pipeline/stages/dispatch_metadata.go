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

	// Get request ID from proto request if available
	var requestID string
	if protoReq, ok := protoRequest.(*ProtoGenerateRequest); ok {
		requestID = protoReq.RequestID
	}
	if requestID == "" {
		requestID = uuid.New().String()
	}

	// Get model ID
	modelID := "unknown"
	if ctx.Input.ModelID != nil {
		modelID = *ctx.Input.ModelID
	}

	// Determine if streaming
	isStreaming := false
	if protoReq, ok := protoRequest.(*ProtoGenerateRequest); ok {
		isStreaming = protoReq.Stream
	}

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
