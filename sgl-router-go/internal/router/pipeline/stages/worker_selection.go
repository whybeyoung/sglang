package stages

import (
	"fmt"

	"github.com/sglang/sglang-router-go/internal/core"
	"github.com/sglang/sglang-router-go/internal/router/pipeline"
	"go.uber.org/zap"
)

// WorkerSelectionMode represents the mode for worker selection
type WorkerSelectionMode int

const (
	WorkerSelectionModeRegular WorkerSelectionMode = iota
	WorkerSelectionModePrefillDecode
)

// WorkerSelectionStage selects appropriate worker(s) based on routing mode
// Similar to Rust WorkerSelectionStage
type WorkerSelectionStage struct {
	*pipeline.BaseStage
	workerRegistry *core.WorkerRegistry
	policyRegistry interface{} // TODO: Define PolicyRegistry interface
	mode           WorkerSelectionMode
}

// NewWorkerSelectionStage creates a new worker selection stage
func NewWorkerSelectionStage(
	workerRegistry *core.WorkerRegistry,
	policyRegistry interface{},
	mode WorkerSelectionMode,
	logger *zap.Logger,
) *WorkerSelectionStage {
	return &WorkerSelectionStage{
		BaseStage:      pipeline.NewBaseStage("WorkerSelection", logger),
		workerRegistry: workerRegistry,
		policyRegistry: policyRegistry,
		mode:           mode,
	}
}

// Execute implements PipelineStage
func (s *WorkerSelectionStage) Execute(ctx *pipeline.RequestContext) (interface{}, error) {
	prep := ctx.State.Preparation
	if prep == nil {
		return nil, fmt.Errorf("preparation stage not completed")
	}

	// Get text for cache-aware selection
	// For Harmony, use selection_text; otherwise use original_text
	var text *string
	if prep.OriginalText != nil {
		text = prep.OriginalText
	}

	var workers *pipeline.WorkerSelection

	switch s.mode {
	case WorkerSelectionModeRegular:
		worker, err := s.selectSingleWorker(ctx.Input.ModelID, text)
		if err != nil {
			return nil, err
		}
		if worker == nil {
			return nil, fmt.Errorf("no available workers for model: %v", ctx.Input.ModelID)
		}
		workers = &pipeline.WorkerSelection{
			IsDual: false,
			Single: worker,
		}

	case WorkerSelectionModePrefillDecode:
		prefill, decode, err := s.selectPDPair(ctx.Input.ModelID, text)
		if err != nil {
			return nil, err
		}
		if prefill == nil || decode == nil {
			return nil, fmt.Errorf("no available PD worker pairs for model: %v", ctx.Input.ModelID)
		}
		workers = &pipeline.WorkerSelection{
			IsDual: true,
		}
		workers.Dual.Prefill = prefill
		workers.Dual.Decode = decode
	}

	ctx.State.Workers = workers

	s.Logger.Debug("Workers selected",
		zap.Bool("is_dual", workers.IsDual),
	)

	return nil, nil // Continue to next stage
}

// selectSingleWorker selects a single worker for regular mode
// Similar to Rust select_single_worker method
func (s *WorkerSelectionStage) selectSingleWorker(modelID *string, text *string) (core.Worker, error) {
	// Get workers for the specified model, filtered by connection mode
	grpcMode := core.ConnectionModeGRPC
	workers := s.workerRegistry.GetWorkersFiltered(
		modelID,
		func() *core.WorkerType { t := core.WorkerTypeRegular; return &t }(),
		&grpcMode,
		false, // get all workers, filter by availability next
	)

	// Filter by availability (health + circuit breaker)
	var available []core.Worker
	for _, w := range workers {
		if w.IsAvailable() {
			available = append(available, w)
		}
	}

	if len(available) == 0 {
		return nil, fmt.Errorf("no available workers")
	}

	// Get the appropriate policy for this model
	// TODO: Get policy from policyRegistry
	// policy := s.policyRegistry.GetPolicyOrDefault(modelID)

	// Select worker using the policy
	// TODO: Implement policy selection
	// idx := policy.SelectWorker(available, text)
	// For now, use first available worker
	// In Rust: idx = policy.select_worker(&available, text)?

	return available[0], nil
}

// selectPDPair selects a prefill-decode worker pair
// Similar to Rust select_pd_pair method
func (s *WorkerSelectionStage) selectPDPair(modelID *string, text *string) (core.Worker, core.Worker, error) {
	grpcMode := core.ConnectionModeGRPC
	allWorkers := s.workerRegistry.GetWorkersFiltered(
		modelID,
		nil, // any worker type
		&grpcMode,
		false,
	)

	// Separate into prefill and decode workers
	var availablePrefill []core.Worker
	var availableDecode []core.Worker

	for _, w := range allWorkers {
		if !w.IsAvailable() {
			continue
		}

		switch w.WorkerType() {
		case core.WorkerTypePrefill:
			availablePrefill = append(availablePrefill, w)
		case core.WorkerTypeDecode:
			availableDecode = append(availableDecode, w)
		}
	}

	if len(availablePrefill) == 0 {
		s.Logger.Warn("No available prefill workers")
		return nil, nil, fmt.Errorf("no available prefill workers")
	}

	if len(availableDecode) == 0 {
		s.Logger.Warn("No available decode workers")
		return nil, nil, fmt.Errorf("no available decode workers")
	}

	// Select using policies
	// TODO: Get policy from policyRegistry and use it for selection
	// For now, use first available
	prefill := availablePrefill[0]
	decode := availableDecode[0]

	return prefill, decode, nil
}
