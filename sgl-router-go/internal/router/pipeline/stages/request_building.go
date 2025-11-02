package stages

import (
	"fmt"

	"github.com/google/uuid"
	"github.com/sglang/sglang-router-go/internal/router/pipeline"
	"github.com/sglang/sglang-router-go/pkg/proto"
	"go.uber.org/zap"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// RequestBuildingStage builds gRPC request messages from prepared context
// Similar to Rust RequestBuildingStage
// NOTE: Full implementation requires proto-generated types
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

	// Generate request ID if not already set
	requestID := uuid.New().String()
	if ctx.State.Dispatch != nil {
		requestID = ctx.State.Dispatch.RequestID
	}

	// Build gRPC GenerateRequest using actual proto types
	// Similar to Rust: proto::GenerateRequest is built from prep.token_ids, sampling params, etc.

	// Build tokenized input
	var originalText string
	if prep.OriginalText != nil {
		originalText = *prep.OriginalText
	}
	tokenizedInput := &proto.TokenizedInput{
		OriginalText: originalText,
		InputIds:     prep.TokenIDs,
	}

	// Build sampling params
	samplingParams := buildSamplingParams(ctx)

	// Determine streaming
	isStreaming := false
	if ctx.State.Dispatch != nil {
		isStreaming = ctx.State.Dispatch.IsStreaming
	}

	// Build proto request
	protoRequest := &proto.GenerateRequest{
		RequestId:      requestID,
		Tokenized:      tokenizedInput,
		SamplingParams: samplingParams,
		Stream:         isStreaming,
		LogMetrics:     true,
		Timestamp:      timestamppb.Now(),
	}

	// Inject PD metadata if needed
	if s.injectPDMetadata {
		// Extract bootstrap info from workers (if in dual mode)
		if ctx.State.Workers != nil && ctx.State.Workers.IsDual {
			// TODO: Implement worker.BootstrapHost() and BootstrapPort() methods
			// For now, placeholder
			// prefillWorker := ctx.State.Workers.Dual.Prefill
			// hostname := prefillWorker.BootstrapHost()
			// port := prefillWorker.BootstrapPort()
			// roomID := generateRandomRoomID()
			// protoRequest.DisaggregatedParams = &proto.DisaggregatedParams{
			//   BootstrapHost: hostname,
			//   BootstrapPort: int32(port),
			//   BootstrapRoom: int32(roomID),
			// }
			s.Logger.Debug("PD metadata injection requested but not yet fully implemented")
		}
	}

	ctx.State.ProtoRequest = protoRequest

	s.Logger.Debug("Request built",
		zap.Bool("inject_pd_metadata", s.injectPDMetadata),
		zap.Int("token_count", len(prep.TokenIDs)),
		zap.String("request_id", requestID),
	)

	return nil, nil // Continue to next stage
}

// buildSamplingParams builds sampling parameters from request context
// Similar to Rust conversion from SamplingParams to proto::SamplingParams
// IMPORTANT: Do not use proto3 defaults (0 for numeric fields)!
// Rust uses explicit defaults (temperature=1.0, top_p=1.0, top_k=-1)
func buildSamplingParams(ctx *pipeline.RequestContext) *proto.SamplingParams {
	// TODO: Extract from actual request (ChatCompletionRequest or GenerateRequest)
	// For now, return sensible defaults based on Rust implementation

	// Use explicit defaults (NOT proto3 defaults)
	params := &proto.SamplingParams{
		Temperature:       1.0, // Explicit default, not 0
		TopP:              1.0, // Explicit default, not 0
		TopK:              -1,  // Explicit default, not 0
		SkipSpecialTokens: true,
		NoStopTrim:        false,
		Stop:              []string{},
		StopTokenIds:      []uint32{},
	}

	// TODO: Extract actual values from ctx.Input.RequestType (ChatCompletionRequest or GenerateRequest)
	// For example:
	// if chatReq, ok := ctx.Input.RequestType.(*protocols.ChatCompletionRequest); ok {
	//     if chatReq.Temperature != nil {
	//         params.Temperature = *chatReq.Temperature
	//     }
	//     // ... etc
	// }

	return params
}

// Note: ProtoGenerateRequest, TokenizedInput, and ProtoSamplingParams placeholders
// have been removed - we now use proto.GenerateRequest, proto.TokenizedInput,
// and proto.SamplingParams directly from the generated proto code.
