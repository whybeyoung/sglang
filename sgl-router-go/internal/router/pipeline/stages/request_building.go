package stages

import (
	"fmt"

	"github.com/google/uuid"
	"github.com/sglang/sglang-router-go/internal/protocols"
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

	// Determine streaming from request or dispatch metadata
	isStreaming := false
	if ctx.State.Dispatch != nil {
		isStreaming = ctx.State.Dispatch.IsStreaming
	} else if ctx.Input.Request != nil {
		// Extract from request if dispatch not yet set
		if chatReq, ok := ctx.Input.Request.(*protocols.ChatCompletionRequest); ok {
			isStreaming = chatReq.Stream
		} else if genReq, ok := ctx.Input.Request.(*protocols.GenerateRequest); ok {
			isStreaming = genReq.Stream
		}
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

	// Extract actual values from request object
	if ctx.Input.Request == nil {
		return params
	}

	switch req := ctx.Input.Request.(type) {
	case *protocols.ChatCompletionRequest:
		// Extract from ChatCompletionRequest
		if req.Temperature != nil {
			params.Temperature = *req.Temperature
		}
		if req.TopP != nil {
			params.TopP = *req.TopP
		}
		if req.TopK != nil {
			params.TopK = *req.TopK
		}
		if req.MaxTokens != nil {
			maxTokens := int32(*req.MaxTokens)
			params.MaxNewTokens = &maxTokens
		} else if req.MaxCompletionTokens != nil {
			maxTokens := int32(*req.MaxCompletionTokens)
			params.MaxNewTokens = &maxTokens
		}
		if req.FrequencyPenalty != nil {
			params.FrequencyPenalty = *req.FrequencyPenalty
		}
		if req.PresencePenalty != nil {
			params.PresencePenalty = *req.PresencePenalty
		}
		if req.LogitBias != nil {
			params.LogitBias = req.LogitBias
		}
		// Handle stop sequences
		if req.Stop != nil {
			if stopStr, ok := req.Stop.(string); ok {
				params.Stop = []string{stopStr}
			} else if stopSlice, ok := req.Stop.([]string); ok {
				params.Stop = stopSlice
			} else if stopSlice, ok := req.Stop.([]interface{}); ok {
				// Convert []interface{} to []string
				for _, v := range stopSlice {
					if str, ok := v.(string); ok {
						params.Stop = append(params.Stop, str)
					}
				}
			}
		}
		if len(req.StopTokenIDs) > 0 {
			params.StopTokenIds = req.StopTokenIDs
		}
		params.SkipSpecialTokens = req.SkipSpecialTokens
		params.NoStopTrim = req.NoStopTrim
		if req.N != nil {
			params.N = int32(*req.N)
		}

	case *protocols.GenerateRequest:
		// Extract from GenerateRequest
		if req.SamplingParams != nil {
			sp := req.SamplingParams
			if sp.Temperature != nil {
				params.Temperature = *sp.Temperature
			}
			if sp.TopP != nil {
				params.TopP = *sp.TopP
			}
			if sp.TopK != nil {
				params.TopK = *sp.TopK
			}
			if sp.MinP != nil {
				params.MinP = *sp.MinP
			}
			if sp.FrequencyPenalty != nil {
				params.FrequencyPenalty = *sp.FrequencyPenalty
			}
			if sp.PresencePenalty != nil {
				params.PresencePenalty = *sp.PresencePenalty
			}
			if sp.RepetitionPenalty != nil {
				params.RepetitionPenalty = *sp.RepetitionPenalty
			}
			if sp.MaxNewTokens != nil {
				params.MaxNewTokens = sp.MaxNewTokens
			}
			if sp.MinNewTokens != nil {
				params.MinNewTokens = *sp.MinNewTokens
			}
			if len(sp.Stop) > 0 {
				params.Stop = sp.Stop
			}
			if len(sp.StopTokenIDs) > 0 {
				params.StopTokenIds = sp.StopTokenIDs
			}
			if sp.SkipSpecialTokens != nil {
				params.SkipSpecialTokens = *sp.SkipSpecialTokens
			}
			if sp.NoStopTrim != nil {
				params.NoStopTrim = *sp.NoStopTrim
			}
			if sp.IgnoreEOS != nil {
				params.IgnoreEos = *sp.IgnoreEOS
			}
			if sp.N != nil {
				params.N = *sp.N
			}
			if sp.LogitBias != nil {
				params.LogitBias = sp.LogitBias
			}
			// Structured generation constraints
			if sp.Regex != nil {
				params.Constraint = &proto.SamplingParams_Regex{Regex: *sp.Regex}
			} else if sp.JSONSchema != nil {
				params.Constraint = &proto.SamplingParams_JsonSchema{JsonSchema: *sp.JSONSchema}
			} else if sp.EBNFGrammar != nil {
				params.Constraint = &proto.SamplingParams_EbnfGrammar{EbnfGrammar: *sp.EBNFGrammar}
			}
			if sp.StreamInterval != nil {
				params.StreamInterval = sp.StreamInterval
			}
		}
	}

	return params
}

// Note: ProtoGenerateRequest, TokenizedInput, and ProtoSamplingParams placeholders
// have been removed - we now use proto.GenerateRequest, proto.TokenizedInput,
// and proto.SamplingParams directly from the generated proto code.
