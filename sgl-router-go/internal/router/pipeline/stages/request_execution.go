package stages

import (
	"fmt"

	"github.com/sglang/sglang-router-go/internal/router/pipeline"
	"go.uber.org/zap"
)

// ExecutionMode represents the execution mode
type ExecutionMode int

const (
	ExecutionModeSingle ExecutionMode = iota
	ExecutionModeDualDispatch
)

// RequestExecutionStage executes requests via gRPC streams
// Similar to Rust RequestExecutionStage
// NOTE: This is a placeholder implementation - full implementation requires:
// 1. gRPC client setup (see internal/grpc/)
// 2. Stream handling
// 3. Cancellation support
type RequestExecutionStage struct {
	*pipeline.BaseStage
	mode ExecutionMode
}

// NewRequestExecutionStage creates a new request execution stage
func NewRequestExecutionStage(mode ExecutionMode, logger *zap.Logger) *RequestExecutionStage {
	return &RequestExecutionStage{
		BaseStage: pipeline.NewBaseStage("RequestExecution", logger),
		mode:      mode,
	}
}

// Execute implements PipelineStage
func (s *RequestExecutionStage) Execute(ctx *pipeline.RequestContext) (interface{}, error) {
	clients := ctx.State.Clients
	if clients == nil {
		return nil, fmt.Errorf("client acquisition stage not completed")
	}

	protoRequest := ctx.State.ProtoRequest
	if protoRequest == nil {
		return nil, fmt.Errorf("request building stage not completed")
	}

	dispatch := ctx.State.Dispatch
	if dispatch == nil {
		return nil, fmt.Errorf("dispatch metadata stage not completed")
	}

	// TODO: Implement gRPC stream execution
	// For single mode:
	//   stream := client.Generate(ctx, protoRequest)
	//   ctx.State.Response.ExecutionResult = &pipeline.ExecutionResult{
	//     IsDual: false,
	//     Single: stream,
	//   }
	//
	// For dual mode:
	//   prefillStream := prefillClient.Generate(ctx, prefillRequest)
	//   decodeStream := decodeClient.Generate(ctx, decodeRequest)
	//   ctx.State.Response.ExecutionResult = &pipeline.ExecutionResult{
	//     IsDual: true,
	//     Dual: struct {
	//       Prefill interface{}
	//       Decode  interface{}
	//     }{
	//       Prefill: prefillStream,
	//       Decode:  decodeStream,
	//     },
	//   }

	// For now, return error indicating not implemented
	return nil, fmt.Errorf("request execution not yet implemented - requires gRPC client setup")

	// If streaming, return early response here
	// Otherwise, continue to response processing
}
