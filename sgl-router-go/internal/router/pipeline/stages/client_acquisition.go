package stages

import (
	"fmt"

	"github.com/sglang/sglang-router-go/internal/grpc"
	"github.com/sglang/sglang-router-go/internal/router/pipeline"
	"go.uber.org/zap"
)

// ClientAcquisitionStage acquires gRPC client connections for selected workers
// Similar to Rust ClientAcquisitionStage
type ClientAcquisitionStage struct {
	*pipeline.BaseStage
	clientPool *grpc.ClientPool
}

// NewClientAcquisitionStage creates a new client acquisition stage
func NewClientAcquisitionStage(clientPool *grpc.ClientPool, logger *zap.Logger) *ClientAcquisitionStage {
	return &ClientAcquisitionStage{
		BaseStage:  pipeline.NewBaseStage("ClientAcquisition", logger),
		clientPool: clientPool,
	}
}

// Execute implements PipelineStage
func (s *ClientAcquisitionStage) Execute(ctx *pipeline.RequestContext) (interface{}, error) {
	workers := ctx.State.Workers
	if workers == nil {
		return nil, fmt.Errorf("worker selection stage not completed")
	}

	// Acquire clients based on worker selection mode
	if workers.IsDual {
		// Dual mode: acquire prefill and decode clients
		prefillConn, err := s.clientPool.GetClient(ctx.Context(), workers.Dual.Prefill)
		if err != nil {
			return nil, fmt.Errorf("failed to acquire prefill client: %w", err)
		}

		decodeConn, err := s.clientPool.GetClient(ctx.Context(), workers.Dual.Decode)
		if err != nil {
			return nil, fmt.Errorf("failed to acquire decode client: %w", err)
		}

		ctx.State.Clients = &pipeline.ClientSelection{
			IsDual: true,
		}
		ctx.State.Clients.Dual.Prefill = prefillConn
		ctx.State.Clients.Dual.Decode = decodeConn
	} else {
		// Single mode: acquire single client
		conn, err := s.clientPool.GetClient(ctx.Context(), workers.Single)
		if err != nil {
			return nil, fmt.Errorf("failed to acquire client: %w", err)
		}

		ctx.State.Clients = &pipeline.ClientSelection{
			IsDual: false,
			Single: conn,
		}
	}

	s.Logger.Debug("Clients acquired",
		zap.Bool("is_dual", workers.IsDual),
	)

	return nil, nil // Continue to next stage
}
