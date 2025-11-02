package core

import (
	"context"
	"sync"
	"time"
)

// ConnectionMode represents how to connect to a worker
type ConnectionMode string

const (
	ConnectionModeHTTP ConnectionMode = "http"
	ConnectionModeGRPC ConnectionMode = "grpc"
)

// WorkerType represents the type of worker
type WorkerType int

const (
	WorkerTypeRegular WorkerType = iota
	WorkerTypePrefill
	WorkerTypeDecode
)

func (wt WorkerType) String() string {
	switch wt {
	case WorkerTypeRegular:
		return "regular"
	case WorkerTypePrefill:
		return "prefill"
	case WorkerTypeDecode:
		return "decode"
	default:
		return "unknown"
	}
}

// WorkerMetadata contains metadata about a worker
type WorkerMetadata struct {
	URL            string
	ModelID        string
	WorkerType     WorkerType
	ConnectionMode ConnectionMode
	Priority       int
	Cost           float64
	Labels         map[string]string
	APIKey         *string
	BootstrapPort  *int // For prefill workers
}

// Worker represents a backend worker that can process requests
// This interface is similar to the Rust Worker trait
type Worker interface {
	// URL returns the worker's connection URL
	URL() string

	// ModelID returns the model ID this worker serves
	ModelID() string

	// WorkerType returns the type of worker (Regular, Prefill, Decode)
	WorkerType() WorkerType

	// ConnectionMode returns how to connect to this worker
	ConnectionMode() ConnectionMode

	// IsHealthy returns whether the worker is currently healthy
	IsHealthy() bool

	// IsAvailable returns whether the worker can accept requests
	// (healthy && circuit breaker not open)
	IsAvailable() bool

	// Load returns the current load metric
	Load() int

	// ResetLoad resets the load counter (periodically called)
	ResetLoad()

	// IncrementLoad increments the load counter
	IncrementLoad()

	// DecrementLoad decrements the load counter
	DecrementLoad()

	// CheckHealth performs a health check
	CheckHealth(ctx context.Context) error

	// CheckHealthAsync performs an async health check (fires and forgets)
	CheckHealthAsync(ctx context.Context)

	// Metadata returns worker metadata
	Metadata() *WorkerMetadata

	// APIKey returns the API key for this worker (if any)
	APIKey() *string
}

// BasicWorker is a basic implementation of the Worker interface
type BasicWorker struct {
	metadata      *WorkerMetadata
	healthy       bool
	load          int
	mu            sync.RWMutex
	lastCheck     time.Time
	checkInterval time.Duration
}

// NewBasicWorker creates a new basic worker
func NewBasicWorker(metadata *WorkerMetadata) *BasicWorker {
	return &BasicWorker{
		metadata:      metadata,
		healthy:       true,
		load:          0,
		checkInterval: 60 * time.Second,
		lastCheck:     time.Now(),
	}
}

func (w *BasicWorker) URL() string {
	return w.metadata.URL
}

func (w *BasicWorker) ModelID() string {
	return w.metadata.ModelID
}

func (w *BasicWorker) WorkerType() WorkerType {
	return w.metadata.WorkerType
}

func (w *BasicWorker) ConnectionMode() ConnectionMode {
	return w.metadata.ConnectionMode
}

func (w *BasicWorker) IsHealthy() bool {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.healthy
}

func (w *BasicWorker) IsAvailable() bool {
	w.mu.RLock()
	defer w.mu.RUnlock()
	// In Rust version, this also checks circuit breaker
	// TODO: Integrate with circuit breaker when implemented
	return w.healthy
}

func (w *BasicWorker) Load() int {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.load
}

func (w *BasicWorker) ResetLoad() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.load = 0
}

func (w *BasicWorker) IncrementLoad() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.load++
}

func (w *BasicWorker) DecrementLoad() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.load > 0 {
		w.load--
	}
}

func (w *BasicWorker) CheckHealth(ctx context.Context) error {
	// TODO: Implement actual health check based on connection mode
	// For gRPC, call HealthCheck RPC
	// For HTTP, call /health endpoint
	w.mu.Lock()
	defer w.mu.Unlock()
	w.lastCheck = time.Now()
	// For now, assume healthy (will be implemented with actual health checks)
	w.healthy = true
	return nil
}

func (w *BasicWorker) CheckHealthAsync(ctx context.Context) {
	go func() {
		_ = w.CheckHealth(ctx)
	}()
}

func (w *BasicWorker) Metadata() *WorkerMetadata {
	return w.metadata
}

func (w *BasicWorker) APIKey() *string {
	return w.metadata.APIKey
}

// SetHealthy updates the health status (used by health checker)
func (w *BasicWorker) SetHealthy(healthy bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.healthy = healthy
	w.lastCheck = time.Now()
}
