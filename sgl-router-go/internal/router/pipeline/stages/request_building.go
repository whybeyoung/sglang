package stages

import (
	"fmt"

	"github.com/sglang/sglang-router-go/internal/router/pipeline"
	"go.uber.org/zap"
)

// RequestBuildingStage builds gRPC request messages from prepared context
// Similar to Rust RequestBuildingStage
type RequestBuildingStage struct {
	*pipeline.BaseStage
	injectPDMetadata bool
}

// NewRequestBuildingStage creates a new request building stage
func NewRequestBuildingStage(injectPDMetadata bool, logger *zap.Logger) *RequestBuildingStage {
	return &RequestBuildingStage{
		BaseStage:        pipeline.NewBaseStage("RequestBuilding", logger),
		injectPDMetadata: injectPDMetadata,
	}
}

// Execute implements PipelineStage
func (s *RequestBuildingStage) Execute(ctx *pipeline.RequestContext) (interface{}, error) {
	prep := ctx.State.Preparation
	if prep == nil {
		return nil, fmt.Errorf("preparation stage not completed")
	}

	clients := ctx.State.Clients
	if clients == nil {
		return nil, fmt.Errorf("client acquisition stage not completed")
	}

	// Build gRPC GenerateRequest
	// Note: In Rust, this creates proto::GenerateRequest
	// TODO: Implement proto request building
	// protoRequest := buildGenerateRequest(prep, s.injectPDMetadata)

	ctx.State.ProtoRequest = nil // TODO: Set protoRequest

	s.Logger.Debug("Request built",
		zap.Bool("inject_pd_metadata", s.injectPDMetadata),
	)

	return nil, nil // Continue to next stage
}
