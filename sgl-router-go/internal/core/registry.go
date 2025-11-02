package core

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// WorkerID is a unique identifier for a worker
type WorkerID string

// WorkerRegistry maintains a registry of all workers
// Similar to Rust WorkerRegistry, using Go's sync primitives
type WorkerRegistry struct {
	workers          map[WorkerID]Worker
	workersByModel   map[string][]WorkerID
	workersByType    map[WorkerType][]WorkerID
	workersByURL     map[string]WorkerID
	mu               sync.RWMutex
	logger           *zap.Logger
	healthCheckStop  chan struct{}
	healthCheckGroup sync.WaitGroup
}

// WorkerRegistryStats contains statistics about the registry
type WorkerRegistryStats struct {
	TotalWorkers   int
	TotalModels    int
	HealthyWorkers int
	TotalLoad      int
	RegularWorkers int
	PrefillWorkers int
	DecodeWorkers  int
}

// NewWorkerRegistry creates a new worker registry
func NewWorkerRegistry(logger *zap.Logger) *WorkerRegistry {
	return &WorkerRegistry{
		workers:         make(map[WorkerID]Worker),
		workersByModel:  make(map[string][]WorkerID),
		workersByType:   make(map[WorkerType][]WorkerID),
		workersByURL:    make(map[string]WorkerID),
		logger:          logger,
		healthCheckStop: make(chan struct{}),
	}
}

// Register registers a new worker in the registry
func (r *WorkerRegistry) Register(worker Worker) WorkerID {
	r.mu.Lock()
	defer r.mu.Unlock()

	url := worker.URL()

	// Check if worker with this URL already exists
	var workerID WorkerID
	if existingID, exists := r.workersByURL[url]; exists {
		workerID = existingID
		// Update existing worker
		r.workers[workerID] = worker
	} else {
		// Create new worker ID
		workerID = WorkerID(fmt.Sprintf("%d", time.Now().UnixNano()))
		r.workers[workerID] = worker
		r.workersByURL[url] = workerID
	}

	// Update indexes
	modelID := worker.ModelID()
	r.workersByModel[modelID] = appendIfNotExists(r.workersByModel[modelID], workerID)

	workerType := worker.WorkerType()
	r.workersByType[workerType] = appendIfNotExists(r.workersByType[workerType], workerID)

	r.logger.Info("Worker registered",
		zap.String("worker_id", string(workerID)),
		zap.String("url", url),
		zap.String("model_id", modelID),
		zap.String("worker_type", workerType.String()),
	)

	return workerID
}

// Remove removes a worker from the registry
func (r *WorkerRegistry) Remove(workerID WorkerID) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	worker, exists := r.workers[workerID]
	if !exists {
		return false
	}

	// Remove from main map
	delete(r.workers, workerID)

	// Remove from URL index
	url := worker.URL()
	delete(r.workersByURL, url)

	// Remove from model index
	modelID := worker.ModelID()
	r.workersByModel[modelID] = removeFromSlice(r.workersByModel[modelID], workerID)
	if len(r.workersByModel[modelID]) == 0 {
		delete(r.workersByModel, modelID)
	}

	// Remove from type index
	workerType := worker.WorkerType()
	r.workersByType[workerType] = removeFromSlice(r.workersByType[workerType], workerID)

	r.logger.Info("Worker removed",
		zap.String("worker_id", string(workerID)),
		zap.String("url", url),
	)

	return true
}

// RemoveByURL removes a worker by its URL
func (r *WorkerRegistry) RemoveByURL(url string) bool {
	r.mu.RLock()
	workerID, exists := r.workersByURL[url]
	r.mu.RUnlock()

	if !exists {
		return false
	}

	return r.Remove(workerID)
}

// Get retrieves a worker by ID
func (r *WorkerRegistry) Get(workerID WorkerID) (Worker, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	worker, exists := r.workers[workerID]
	return worker, exists
}

// GetByURL retrieves a worker by URL
func (r *WorkerRegistry) GetByURL(url string) (Worker, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	workerID, exists := r.workersByURL[url]
	if !exists {
		return nil, false
	}
	worker, exists := r.workers[workerID]
	return worker, exists
}

// GetByModel retrieves all workers for a specific model
func (r *WorkerRegistry) GetByModel(modelID string) []Worker {
	r.mu.RLock()
	defer r.mu.RUnlock()

	workerIDs, exists := r.workersByModel[modelID]
	if !exists {
		return nil
	}

	workers := make([]Worker, 0, len(workerIDs))
	for _, id := range workerIDs {
		if worker, exists := r.workers[id]; exists {
			workers = append(workers, worker)
		}
	}

	return workers
}

// GetByType retrieves all workers of a specific type
func (r *WorkerRegistry) GetByType(workerType WorkerType) []Worker {
	r.mu.RLock()
	defer r.mu.RUnlock()

	workerIDs, exists := r.workersByType[workerType]
	if !exists {
		return nil
	}

	workers := make([]Worker, 0, len(workerIDs))
	for _, id := range workerIDs {
		if worker, exists := r.workers[id]; exists {
			workers = append(workers, worker)
		}
	}

	return workers
}

// GetPrefillWorkers retrieves all prefill workers
func (r *WorkerRegistry) GetPrefillWorkers() []Worker {
	return r.GetByType(WorkerTypePrefill)
}

// GetDecodeWorkers retrieves all decode workers
func (r *WorkerRegistry) GetDecodeWorkers() []Worker {
	return r.GetByType(WorkerTypeDecode)
}

// GetAll retrieves all workers
func (r *WorkerRegistry) GetAll() []Worker {
	r.mu.RLock()
	defer r.mu.RUnlock()

	workers := make([]Worker, 0, len(r.workers))
	for _, worker := range r.workers {
		workers = append(workers, worker)
	}

	return workers
}

// GetModels returns all model IDs that have workers
func (r *WorkerRegistry) GetModels() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	models := make([]string, 0, len(r.workersByModel))
	for modelID, workerIDs := range r.workersByModel {
		if len(workerIDs) > 0 {
			models = append(models, modelID)
		}
	}

	return models
}

// GetWorkersFiltered retrieves workers filtered by multiple criteria
// Similar to Rust get_workers_filtered method
func (r *WorkerRegistry) GetWorkersFiltered(
	modelID *string,
	workerType *WorkerType,
	connectionMode *ConnectionMode,
	healthyOnly bool,
) []Worker {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var candidates []Worker

	// Start with most efficient collection based on filters
	if modelID != nil {
		workerIDs, exists := r.workersByModel[*modelID]
		if !exists {
			return nil
		}
		for _, id := range workerIDs {
			if worker, exists := r.workers[id]; exists {
				candidates = append(candidates, worker)
			}
		}
	} else {
		// Get all workers
		for _, worker := range r.workers {
			candidates = append(candidates, worker)
		}
	}

	// Apply filters
	var filtered []Worker
	for _, worker := range candidates {
		// Check worker type
		if workerType != nil && worker.WorkerType() != *workerType {
			continue
		}

		// Check connection mode
		if connectionMode != nil && worker.ConnectionMode() != *connectionMode {
			continue
		}

		// Check health
		if healthyOnly && !worker.IsHealthy() {
			continue
		}

		filtered = append(filtered, worker)
	}

	return filtered
}

// Stats returns statistics about the registry
func (r *WorkerRegistry) Stats() WorkerRegistryStats {
	r.mu.RLock()
	defer r.mu.RUnlock()

	stats := WorkerRegistryStats{
		TotalWorkers: len(r.workers),
		TotalModels:  len(r.workersByModel),
	}

	for _, worker := range r.workers {
		if worker.IsHealthy() {
			stats.HealthyWorkers++
		}
		stats.TotalLoad += worker.Load()

		switch worker.WorkerType() {
		case WorkerTypeRegular:
			stats.RegularWorkers++
		case WorkerTypePrefill:
			stats.PrefillWorkers++
		case WorkerTypeDecode:
			stats.DecodeWorkers++
		}
	}

	return stats
}

// StartHealthChecker starts a background health checker
func (r *WorkerRegistry) StartHealthChecker(ctx context.Context, checkInterval time.Duration) {
	r.healthCheckGroup.Add(1)
	go func() {
		defer r.healthCheckGroup.Done()

		ticker := time.NewTicker(checkInterval)
		defer ticker.Stop()

		checkCount := uint64(0)
		const loadResetInterval = 10

		for {
			select {
			case <-ctx.Done():
				return
			case <-r.healthCheckStop:
				return
			case <-ticker.C:
				// Get all workers
				workers := r.GetAll()

				// Perform health checks
				for _, worker := range workers {
					worker.CheckHealthAsync(ctx)
				}

				// Reset loads periodically (similar to Rust implementation)
				checkCount++
				if checkCount%loadResetInterval == 0 {
					r.logger.Debug("Resetting worker loads", zap.Uint64("cycle", checkCount))
					for _, worker := range workers {
						worker.ResetLoad()
					}
				}
			}
		}
	}()
}

// StopHealthChecker stops the health checker
func (r *WorkerRegistry) StopHealthChecker() {
	close(r.healthCheckStop)
	r.healthCheckGroup.Wait()
}

// Helper functions

func appendIfNotExists(slice []WorkerID, id WorkerID) []WorkerID {
	for _, existingID := range slice {
		if existingID == id {
			return slice
		}
	}
	return append(slice, id)
}

func removeFromSlice(slice []WorkerID, id WorkerID) []WorkerID {
	for i, existingID := range slice {
		if existingID == id {
			return append(slice[:i], slice[i+1:]...)
		}
	}
	return slice
}
