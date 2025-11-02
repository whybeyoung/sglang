package stages

import (
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/sglang/sglang-router-go/internal/router/pipeline"
	"github.com/sglang/sglang-router-go/pkg/proto"
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
	if protoReq, ok := protoRequest.(*proto.GenerateRequest); ok {
		requestID = protoReq.RequestId
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
	if protoReq, ok := protoRequest.(*proto.GenerateRequest); ok {
		isStreaming = protoReq.Stream
	}

	// Extract weight_version from selected worker
	// In Rust: worker.get_model_info() or from metadata
	var weightVersion *string
	if ctx.State.Workers != nil {
		var selectedWorker pipeline.Worker
		if ctx.State.Workers.IsDual {
			// Use prefill worker for weight version (similar to Rust)
			selectedWorker = ctx.State.Workers.Dual.Prefill
		} else {
			selectedWorker = ctx.State.Workers.Single
		}

		if selectedWorker != nil {
			// TODO: Get weight_version from worker via gRPC GetModelInfo call
			// For now, try to get from Labels in metadata
			metadata := selectedWorker.Metadata()
			if metadata != nil && metadata.Labels != nil {
				if wv, ok := metadata.Labels["weight_version"]; ok {
					weightVersion = &wv
				}
			}
		}
	}

	ctx.State.Dispatch = &pipeline.DispatchMetadata{
		RequestID:     requestID,
		Model:         modelID,
		Created:       time.Now().Unix(),
		WeightVersion: weightVersion,
		IsStreaming:   isStreaming,
	}

	s.Logger.Debug("Dispatch metadata prepared",
		zap.String("request_id", requestID),
		zap.String("model", modelID),
		zap.Bool("is_streaming", isStreaming),
	)

	return nil, nil // Continue to next stage
}
