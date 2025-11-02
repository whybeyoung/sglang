package stages

import (
	"context"
	"fmt"
	"sync"

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
// Similar to Rust execute_single method
func (s *RequestExecutionStage) executeSingle(
	ctx context.Context,
	clients *pipeline.ClientSelection,
	grpcRequest interface{}, // TODO: Use actual proto.GenerateRequest
) (*pipeline.ExecutionResult, error) {
	if clients.IsDual {
		return nil, fmt.Errorf("expected single client but got dual")
	}

	clientConn, ok := clients.Single.(*grpc.ClientConn)
	if !ok {
		return nil, fmt.Errorf("invalid client type: expected *grpc.ClientConn, got %T", clients.Single)
	}

	// TODO: Replace with actual proto client call after make generate
	// After proto generation:
	//   client := proto.NewSglangSchedulerClient(clientConn)
	//   stream, err := client.Generate(ctx, grpcRequest.(*proto.GenerateRequest))
	//   if err != nil {
	//       return nil, fmt.Errorf("failed to start generation: %w", err)
	//   }
	//   return &pipeline.ExecutionResult{
	//       IsDual: false,
	//       Single: stream,
	//   }, nil

	// For now, log and return placeholder
	requestID := uuid.New().String()
	s.Logger.Warn("gRPC Generate call using placeholder - requires proto-generated client",
		zap.String("request_id", requestID),
		zap.String("connection_state", clientConn.GetState().String()),
	)

	// Create a placeholder stream
	// In actual implementation, this would be the gRPC stream from client.Generate()
	stream := &GRPCStreamPlaceholder{
		RequestID: requestID,
		ctx:       ctx,
	}

	return &pipeline.ExecutionResult{
		IsDual: false,
		Single: stream,
	}, nil
}

// executeDual executes a dual-worker (PD) request
// Similar to Rust execute_dual_dispatch method
// In PD mode, we dispatch to both prefill and decode workers in parallel
func (s *RequestExecutionStage) executeDual(
	ctx context.Context,
	clients *pipeline.ClientSelection,
	grpcRequest interface{},
) (*pipeline.ExecutionResult, error) {
	if !clients.IsDual {
		return nil, fmt.Errorf("expected dual clients but got single")
	}

	prefillConn, ok1 := clients.Dual.Prefill.(*grpc.ClientConn)
	decodeConn, ok2 := clients.Dual.Decode.(*grpc.ClientConn)
	if !ok1 || !ok2 {
		return nil, fmt.Errorf("invalid client types: expected *grpc.ClientConn, got prefill=%T, decode=%T",
			clients.Dual.Prefill, clients.Dual.Decode)
	}

	// TODO: Replace with actual proto client calls after make generate
	// After proto generation:
	//   prefillClient := proto.NewSglangSchedulerClient(prefillConn)
	//   decodeClient := proto.NewSglangSchedulerClient(decodeConn)
	//
	//   // Clone request for both workers (similar to Rust)
	//   prefillReq := grpcRequest.(*proto.GenerateRequest) // Assuming Clone() method exists
	//   decodeReq := prefillReq.Clone()
	//
	//   // Execute both in parallel (similar to Rust tokio::join!)
	//   var prefillStream, decodeStream GenerateStream
	//   var prefillErr, decodeErr error
	//   var wg sync.WaitGroup
	//   wg.Add(2)
	//   go func() {
	//       defer wg.Done()
	//       prefillStream, prefillErr = prefillClient.Generate(ctx, prefillReq)
	//   }()
	//   go func() {
	//       defer wg.Done()
	//       decodeStream, decodeErr = decodeClient.Generate(ctx, decodeReq)
	//   }()
	//   wg.Wait()
	//
	//   if prefillErr != nil {
	//       return nil, fmt.Errorf("prefill worker failed to start: %w", prefillErr)
	//   }
	//   if decodeErr != nil {
	//       return nil, fmt.Errorf("decode worker failed to start: %w", decodeErr)
	//   }
	//
	//   return &pipeline.ExecutionResult{
	//       IsDual: true,
	//       Dual: struct {
	//           Prefill interface{}
	//           Decode  interface{}
	//       }{
	//           Prefill: prefillStream,
	//           Decode:  decodeStream,
	//       },
	//   }, nil

	// For now, log and return placeholder
	s.Logger.Warn("gRPC dual dispatch using placeholder - requires proto-generated client",
		zap.String("prefill_state", prefillConn.GetState().String()),
		zap.String("decode_state", decodeConn.GetState().String()),
	)

	prefillStream := &GRPCStreamPlaceholder{
		RequestID: "prefill",
		ctx:       ctx,
	}
	decodeStream := &GRPCStreamPlaceholder{
		RequestID: "decode",
		ctx:       ctx,
	}

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
// This should implement the GenerateStream interface from grpc/proto_interface.go
type GRPCStreamPlaceholder struct {
	RequestID string
	ctx       context.Context
	mu        sync.RWMutex
	closed    bool
}

// Recv is a placeholder for receiving chunks from the stream
func (g *GRPCStreamPlaceholder) Recv() (interface{}, error) {
	g.mu.RLock()
	if g.closed {
		g.mu.RUnlock()
		return nil, fmt.Errorf("stream closed")
	}
	g.mu.RUnlock()

	// Wait for context cancellation
	<-g.ctx.Done()
	return nil, g.ctx.Err()
}

// Context returns the stream context
func (g *GRPCStreamPlaceholder) Context() context.Context {
	return g.ctx
}

// CloseSend closes the send direction (placeholder)
func (g *GRPCStreamPlaceholder) CloseSend() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.closed = true
	return nil
}
