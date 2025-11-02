package policy

import (
	"github.com/sglang/sglang-router-go/internal/core"
)

// Policy represents a load balancing policy
// Similar to Rust Policy trait
type Policy interface {
	// SelectWorker selects a worker index from the available workers
	// text can be used for cache-aware policies
	SelectWorker(workers []core.Worker, text *string) (int, bool)

	// Name returns the policy name
	Name() string
}

// PolicyRegistry maintains a registry of policies
// Similar to Rust PolicyRegistry
type PolicyRegistry struct {
	policies      map[string]Policy
	defaultPolicy Policy
}

// NewPolicyRegistry creates a new policy registry
func NewPolicyRegistry(defaultPolicy Policy) *PolicyRegistry {
	return &PolicyRegistry{
		policies:      make(map[string]Policy),
		defaultPolicy: defaultPolicy,
	}
}

// Register registers a policy for a specific model
func (r *PolicyRegistry) Register(modelID string, policy Policy) {
	r.policies[modelID] = policy
}

// GetPolicyOrDefault gets the policy for a model, or returns the default
func (r *PolicyRegistry) GetPolicyOrDefault(modelID string) Policy {
	if policy, exists := r.policies[modelID]; exists {
		return policy
	}
	return r.defaultPolicy
}

// GetDefaultPolicy returns the default policy
func (r *PolicyRegistry) GetDefaultPolicy() Policy {
	return r.defaultPolicy
}
