package stages

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/sglang/sglang-router-go/internal/router/pipeline"
	"go.uber.org/zap"
	"google.golang.org/grpc"
)

// ExecutionMode represents the execution mode
type ExecutionMode int

const (
	ExecutionModeSingle ExecutionMode = iota
	ExecutionModeDualDispatch
)

// RequestExecutionStage executes requests via gRPC streams
// Similar to Rust RequestExecutionStage
// NOTE: Full implementation requires proto-generated gRPC client stub
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
		// Generate request ID if not set
		requestID := uuid.New().String()
		ctx.State.Dispatch = &pipeline.DispatchMetadata{
			RequestID:   requestID,
			Model:       "default",
			Created:     0,
			IsStreaming: false,
		}
		dispatch = ctx.State.Dispatch
	}

	// Convert proto request to actual proto type
	// TODO: Replace with actual proto conversion after make generate
	grpcRequest, err := s.convertToGRPCRequest(protoRequest)
	if err != nil {
		return nil, fmt.Errorf("failed to convert request: %w", err)
	}

	// Execute based on mode
	var result *pipeline.ExecutionResult
	if s.mode == ExecutionModeSingle {
		result, err = s.executeSingle(ctx.Context(), clients, grpcRequest)
	} else {
		result, err = s.executeDual(ctx.Context(), clients, grpcRequest)
	}

	if err != nil {
		return nil, fmt.Errorf("execution failed: %w", err)
	}

	ctx.State.Response.ExecutionResult = result

	s.Logger.Debug("Request executed",
		zap.Bool("is_dual", result.IsDual),
		zap.Bool("is_streaming", dispatch.IsStreaming),
	)

	return nil, nil // Continue to response processing
}

// executeSingle executes a single-worker request
func (s *RequestExecutionStage) executeSingle(
	ctx context.Context,
	clients *pipeline.ClientSelection,
	grpcRequest interface{}, // TODO: Use actual proto.GenerateRequest
) (*pipeline.ExecutionResult, error) {
	if clients.IsDual {
		return nil, fmt.Errorf("expected single client but got dual")
	}

	_, ok := clients.Single.(*grpc.ClientConn)
	if !ok {
		return nil, fmt.Errorf("invalid client type: expected *grpc.ClientConn")
	}

	// TODO: Replace with actual proto client call after make generate
	// conn := clients.Single.(*grpc.ClientConn)
	// client := proto.NewSglangSchedulerClient(conn)
	// stream, err := client.Generate(ctx, grpcRequest.(*proto.GenerateRequest))

	// Generate request ID for placeholder
	requestID := uuid.New().String()

	// For now, return placeholder
	s.Logger.Warn("gRPC Generate call not implemented - requires proto-generated client",
		zap.String("request_id", requestID),
	)

	// Create a placeholder stream
	// In actual implementation, this would be the gRPC stream from client.Generate()
	stream := &GRPCStreamPlaceholder{
		RequestID: requestID,
	}

	return &pipeline.ExecutionResult{
		IsDual: false,
		Single: stream,
	}, nil
}

// executeDual executes a dual-worker (PD) request
func (s *RequestExecutionStage) executeDual(
	ctx context.Context,
	clients *pipeline.ClientSelection,
	grpcRequest interface{},
) (*pipeline.ExecutionResult, error) {
	if !clients.IsDual {
		return nil, fmt.Errorf("expected dual clients but got single")
	}

	_, ok1 := clients.Dual.Prefill.(*grpc.ClientConn)
	_, ok2 := clients.Dual.Decode.(*grpc.ClientConn)
	if !ok1 || !ok2 {
		return nil, fmt.Errorf("invalid client types: expected *grpc.ClientConn")
	}

	// TODO: Replace with actual proto client calls
	// prefillConn := clients.Dual.Prefill.(*grpc.ClientConn)
	// decodeConn := clients.Dual.Decode.(*grpc.ClientConn)
	// prefillClient := proto.NewSglangSchedulerClient(prefillConn)
	// decodeClient := proto.NewSglangSchedulerClient(decodeConn)
	//
	// prefillStream, err := prefillClient.Generate(ctx, grpcRequest)
	// decodeStream, err := decodeClient.Generate(ctx, grpcRequest)

	s.Logger.Warn("gRPC dual dispatch not implemented - requires proto-generated client")

	prefillStream := &GRPCStreamPlaceholder{RequestID: "prefill"}
	decodeStream := &GRPCStreamPlaceholder{RequestID: "decode"}

	return &pipeline.ExecutionResult{
		IsDual: true,
		Dual: struct {
			Prefill interface{}
			Decode  interface{}
		}{
			Prefill: prefillStream,
			Decode:  decodeStream,
		},
	}, nil
}

// convertToGRPCRequest converts our placeholder proto request to actual proto type
// TODO: Replace with actual conversion after proto code generation
func (s *RequestExecutionStage) convertToGRPCRequest(req interface{}) (interface{}, error) {
	// For now, just pass through
	// After proto generation, convert ProtoGenerateRequest to proto.GenerateRequest
	return req, nil
}

// GRPCStreamPlaceholder is a placeholder for the actual gRPC stream
// TODO: Replace with actual proto stream type after make generate
type GRPCStreamPlaceholder struct {
	RequestID string
}
