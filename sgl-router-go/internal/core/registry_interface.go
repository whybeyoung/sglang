package core

// RegistryAccessor provides an interface for accessing worker registry
// This helps avoid import cycles between core and server packages
type RegistryAccessor interface {
	GetAll() []Worker
	Stats() WorkerRegistryStats
	GetByModel(modelID string) []Worker
	GetByType(workerType WorkerType) []Worker
	Register(worker Worker) WorkerID
	Remove(workerID WorkerID) bool
	Get(workerID WorkerID) (Worker, bool)
	GetByURL(url string) (Worker, bool)
}
