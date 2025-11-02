package core

import (
	"strings"

	"go.uber.org/zap"
)

// RegistryAdapter adapts WorkerRegistry to RegistryAccessor interface
// This allows server package to use registry without import cycles
type RegistryAdapter struct {
	registry *WorkerRegistry
	logger   *zap.Logger
}

// NewRegistryAdapter creates a new registry adapter
func NewRegistryAdapter(registry *WorkerRegistry, logger *zap.Logger) *RegistryAdapter {
	return &RegistryAdapter{
		registry: registry,
		logger:   logger,
	}
}

// GetAll returns all workers as Worker interface
func (a *RegistryAdapter) GetAll() []interface{} {
	workers := a.registry.GetAll()
	result := make([]interface{}, len(workers))
	for i := range workers {
		// Create a serializable wrapper
		result[i] = &workerInfo{
			URL:            workers[i].URL(),
			ModelID:        workers[i].ModelID(),
			WorkerType:     workers[i].WorkerType().String(),
			ConnectionMode: workers[i].ConnectionMode(),
			Healthy:        workers[i].IsHealthy(),
			Load:           workers[i].Load(),
		}
	}
	return result
}

// workerInfo is a serializable representation of a worker
type workerInfo struct {
	URL            string
	ModelID        string
	WorkerType     string
	ConnectionMode ConnectionMode
	Healthy        bool
	Load           int
}

// Stats returns statistics
func (a *RegistryAdapter) Stats() interface{} {
	return a.registry.Stats()
}

// GetByModel returns workers by model ID
func (a *RegistryAdapter) GetByModel(modelID string) []interface{} {
	workers := a.registry.GetByModel(modelID)
	result := make([]interface{}, len(workers))
	for i, w := range workers {
		result[i] = w
	}
	return result
}

// RegisterWorker registers a new worker from API request
func (a *RegistryAdapter) RegisterWorker(url, modelID, workerType, connectionMode string) error {
	// Parse connection mode
	var connMode ConnectionMode
	switch strings.ToLower(connectionMode) {
	case "http":
		connMode = ConnectionModeHTTP
	case "grpc":
		connMode = ConnectionModeGRPC
	default:
		// Try to infer from URL
		if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
			connMode = ConnectionModeHTTP
		} else {
			connMode = ConnectionModeGRPC
		}
	}

	// Parse worker type
	var wt WorkerType
	switch strings.ToLower(workerType) {
	case "prefill":
		wt = WorkerTypePrefill
	case "decode":
		wt = WorkerTypeDecode
	default:
		wt = WorkerTypeRegular
	}

	// Create metadata
	metadata := &WorkerMetadata{
		URL:            url,
		ModelID:        modelID,
		WorkerType:     wt,
		ConnectionMode: connMode,
		APIKey:         nil,
		Priority:       50,
		Cost:           1.0,
		Labels:         make(map[string]string),
	}

	// Create worker
	worker := NewBasicWorkerWithLogger(metadata, a.logger)

	// Register
	a.registry.Register(worker)

	return nil
}
