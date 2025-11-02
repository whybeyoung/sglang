package stages

import (
	"context"
	"fmt"
	"io"

	"github.com/sglang/sglang-router-go/pkg/proto"
	"go.uber.org/zap"
)

// collectStreamChunks collects all GenerateComplete responses from a gRPC stream
// Similar to Rust utils::collect_stream_responses
// This processes the stream and collects all Complete responses, ignoring Chunk responses
func collectStreamChunks(
	ctx context.Context,
	stream proto.SglangScheduler_GenerateClient,
	logger *zap.Logger,
) ([]*proto.GenerateComplete, error) {
	var allResponses []*proto.GenerateComplete

	// Main loop: receive responses until stream ends
	for {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Receive next response from stream
		genResponse, err := stream.Recv()
		if err == io.EOF {
			// Stream ended normally
			break
		}
		if err != nil {
			return nil, fmt.Errorf("stream receive error: %w", err)
		}

		// Process response based on type
		if chunk := genResponse.GetChunk(); chunk != nil {
			// Chunk responses are ignored for non-streaming collection
			// We only care about Complete responses
			logger.Debug("Ignoring chunk in non-streaming collection",
				zap.Uint32("index", chunk.Index),
			)
			continue

		} else if complete := genResponse.GetComplete(); complete != nil {
			// Complete response: collect it
			allResponses = append(allResponses, complete)
			logger.Debug("Collected complete response",
				zap.Uint32("index", complete.Index),
				zap.String("finish_reason", complete.FinishReason),
				zap.Int("token_count", len(complete.OutputIds)),
			)

		} else if errResp := genResponse.GetError(); errResp != nil {
			// Error response: return error
			return nil, fmt.Errorf("stream error: %s", errResp.Message)
		}
	}

	if len(allResponses) == 0 {
		return nil, fmt.Errorf("no complete responses received from stream")
	}

	logger.Debug("Stream collection complete",
		zap.Int("response_count", len(allResponses)),
	)

	return allResponses, nil
}
