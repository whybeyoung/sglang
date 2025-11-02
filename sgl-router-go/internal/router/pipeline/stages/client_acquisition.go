package stages

import (
	"fmt"

	"github.com/sglang/sglang-router-go/internal/router/pipeline"
	"go.uber.org/zap"
)

// ClientAcquisitionStage acquires gRPC client connections for selected workers
// Similar to Rust ClientAcquisitionStage
type ClientAcquisitionStage struct {
	*pipeline.BaseStage
	// TODO: Add client pool or factory
}

// NewClientAcquisitionStage creates a new client acquisition stage
func NewClientAcquisitionStage(logger *zap.Logger) *ClientAcquisitionStage {
	return &ClientAcquisitionStage{
		BaseStage: pipeline.NewBaseStage("ClientAcquisition", logger),
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
		// TODO: Implement gRPC client acquisition
		// prefillClient := acquireClient(workers.Dual.Prefill)
		// decodeClient := acquireClient(workers.Dual.Decode)

		ctx.State.Clients = &pipeline.ClientSelection{
			IsDual: true,
			// Dual.Prefill = prefillClient
			// Dual.Decode = decodeClient
		}
	} else {
		// Single mode: acquire single client
		// TODO: Implement gRPC client acquisition
		// client := acquireClient(workers.Single)

		ctx.State.Clients = &pipeline.ClientSelection{
			IsDual: false,
			// Single = client
		}
	}

	s.Logger.Debug("Clients acquired",
		zap.Bool("is_dual", workers.IsDual),
	)

	return nil, nil // Continue to next stage
}
