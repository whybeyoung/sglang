package stages

import (
	"context"
	"fmt"
	"sync"

	"github.com/google/uuid"
	"github.com/sglang/sglang-router-go/internal/grpc"
	"github.com/sglang/sglang-router-go/internal/router/pipeline"
	"github.com/sglang/sglang-router-go/pkg/proto"
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
	grpcRequest, ok := protoRequest.(*proto.GenerateRequest)
	if !ok {
		return nil, fmt.Errorf("invalid proto request type: expected *proto.GenerateRequest, got %T", protoRequest)
	}

	// Execute based on mode
	var result *pipeline.ExecutionResult
	var err error
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
	grpcRequest *proto.GenerateRequest,
) (*pipeline.ExecutionResult, error) {
	if clients.IsDual {
		return nil, fmt.Errorf("expected single client but got dual")
	}

	clientWrapper, ok := clients.Single.(*grpc.SglangSchedulerClientWrapper)
	if !ok {
		return nil, fmt.Errorf("invalid client type: expected *grpc.SglangSchedulerClientWrapper, got %T", clients.Single)
	}

	// Call Generate RPC using proto client
	stream, err := clientWrapper.Generate(ctx, grpcRequest)
	if err != nil {
		return nil, fmt.Errorf("failed to start generation: %w", err)
	}

	s.Logger.Debug("gRPC Generate stream started",
		zap.String("request_id", grpcRequest.RequestId),
	)

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
	grpcRequest *proto.GenerateRequest,
) (*pipeline.ExecutionResult, error) {
	if !clients.IsDual {
		return nil, fmt.Errorf("expected dual clients but got single")
	}

	prefillWrapper, ok1 := clients.Dual.Prefill.(*grpc.SglangSchedulerClientWrapper)
	decodeWrapper, ok2 := clients.Dual.Decode.(*grpc.SglangSchedulerClientWrapper)
	if !ok1 || !ok2 {
		return nil, fmt.Errorf("invalid client types: expected *grpc.SglangSchedulerClientWrapper, got prefill=%T, decode=%T",
			clients.Dual.Prefill, clients.Dual.Decode)
	}

	// Clone request for both workers (similar to Rust)
	// Note: proto messages don't have Clone() by default, we need to manually copy
	prefillReq := s.cloneGenerateRequest(grpcRequest)
	decodeReq := s.cloneGenerateRequest(grpcRequest)

	// Execute both in parallel (similar to Rust tokio::join!)
	var prefillStream proto.SglangScheduler_GenerateClient
	var decodeStream proto.SglangScheduler_GenerateClient
	var prefillErr, decodeErr error

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		prefillStream, prefillErr = prefillWrapper.Generate(ctx, prefillReq)
	}()

	go func() {
		defer wg.Done()
		decodeStream, decodeErr = decodeWrapper.Generate(ctx, decodeReq)
	}()

	wg.Wait()

	if prefillErr != nil {
		return nil, fmt.Errorf("prefill worker failed to start: %w", prefillErr)
	}
	if decodeErr != nil {
		return nil, fmt.Errorf("decode worker failed to start: %w", decodeErr)
	}

	s.Logger.Debug("gRPC dual dispatch streams started",
		zap.String("request_id", grpcRequest.RequestId),
	)

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

// cloneGenerateRequest creates a deep copy of GenerateRequest
// Note: proto messages don't have Clone() by default, so we manually copy fields
func (s *RequestExecutionStage) cloneGenerateRequest(req *proto.GenerateRequest) *proto.GenerateRequest {
	// Use proto.Clone if available, otherwise manual copy
	// For now, create a new request with same fields
	clone := &proto.GenerateRequest{
		RequestId:            req.RequestId,
		Tokenized:            req.Tokenized,
		MmInputs:             req.MmInputs,
		SamplingParams:       req.SamplingParams,
		ReturnLogprob:        req.ReturnLogprob,
		LogprobStartLen:      req.LogprobStartLen,
		TopLogprobsNum:       req.TopLogprobsNum,
		TokenIdsLogprob:      append([]uint32(nil), req.TokenIdsLogprob...),
		ReturnHiddenStates:   req.ReturnHiddenStates,
		DisaggregatedParams:  req.DisaggregatedParams,
		CustomLogitProcessor: req.CustomLogitProcessor,
		Timestamp:            req.Timestamp,
		LogMetrics:           req.LogMetrics,
		InputEmbeds:          append([]float32(nil), req.InputEmbeds...),
		LoraId:               req.LoraId,
		DataParallelRank:     req.DataParallelRank,
		Stream:               req.Stream,
	}
	return clone
}

// convertToGRPCRequest is no longer needed - we now use proto.GenerateRequest directly

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
