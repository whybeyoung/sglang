package stages

import (
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/sglang/sglang-router-go/internal/router/pipeline"
	"go.uber.org/zap"
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

	// Build gRPC GenerateRequest
	// NOTE: This is a placeholder - full implementation requires proto-generated types
	// In Rust: proto::GenerateRequest is built from prep.token_ids, sampling params, etc.

	protoRequest := &ProtoGenerateRequest{
		RequestID: requestID,
		TokenizedInput: &TokenizedInput{
			OriginalText: func() string {
				if prep.OriginalText != nil {
					return *prep.OriginalText
				}
				return ""
			}(),
			InputIDs: prep.TokenIDs,
		},
		SamplingParams: buildSamplingParams(ctx),
		Stream: func() bool {
			if ctx.State.Dispatch != nil {
				return ctx.State.Dispatch.IsStreaming
			}
			return false
		}(),
		LogMetrics: true,
	}

	// Inject PD metadata if needed
	if s.injectPDMetadata {
		// TODO: Extract bootstrap info from workers
		// In Rust: prefill_worker.bootstrap_host() and bootstrap_port()
		// protoRequest.DisaggregatedParams = &DisaggregatedParams{
		//   BootstrapHost: hostname,
		//   BootstrapPort: port,
		//   BootstrapRoom: roomID,
		// }
		s.Logger.Debug("PD metadata injection requested but not yet implemented")
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
func buildSamplingParams(ctx *pipeline.RequestContext) *ProtoSamplingParams {
	// TODO: Extract from actual request (ChatCompletionRequest or GenerateRequest)
	// For now, return sensible defaults based on Rust implementation
	// Note: Rust uses explicit defaults (temperature=1.0, top_p=1.0, top_k=-1)
	// NOT proto3 defaults (0 for numeric fields)

	temp := float32(1.0)
	topP := float32(1.0)
	topK := int32(-1)

	return &ProtoSamplingParams{
		Temperature:       &temp,
		TopP:              &topP,
		TopK:              &topK,
		MaxNewTokens:      nil, // Will be set from request
		Stop:              []string{},
		StopTokenIDs:      []uint32{},
		SkipSpecialTokens: func() *bool { b := true; return &b }(),
		NoStopTrim:        func() *bool { b := false; return &b }(),
	}
}

// ProtoGenerateRequest is a placeholder for the actual proto-generated type
// TODO: Replace with actual proto.GenerateRequest after running make generate
type ProtoGenerateRequest struct {
	RequestID      string
	TokenizedInput *TokenizedInput
	SamplingParams *ProtoSamplingParams
	Stream         bool
	LogMetrics     bool
	Created        time.Time
	// TODO: Add other fields from proto (MultimodalInputs, DisaggregatedParams, etc.)
}

// TokenizedInput represents tokenized input
// TODO: Replace with actual proto.TokenizedInput
type TokenizedInput struct {
	OriginalText string
	InputIDs     []uint32
}

// ProtoSamplingParams is a placeholder for proto.SamplingParams
// TODO: Replace with actual proto.SamplingParams
// NOTE: Using pointers to distinguish unset vs zero values (similar to proto3 optional)
type ProtoSamplingParams struct {
	Temperature       *float32
	TopP              *float32
	TopK              *int32
	MaxNewTokens      *int32
	Stop              []string
	StopTokenIDs      []uint32
	SkipSpecialTokens *bool
	NoStopTrim        *bool
	// TODO: Add other fields (FrequencyPenalty, PresencePenalty, RepetitionPenalty, etc.)
}
