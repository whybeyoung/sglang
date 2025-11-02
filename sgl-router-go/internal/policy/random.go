package policy

import (
	"math/rand"

	"github.com/sglang/sglang-router-go/internal/core"
)

// RandomPolicy implements random worker selection
// Similar to Rust RandomPolicy
type RandomPolicy struct{}

// NewRandomPolicy creates a new random policy
func NewRandomPolicy() *RandomPolicy {
	return &RandomPolicy{}
}

// SelectWorker selects a random worker
func (p *RandomPolicy) SelectWorker(workers []core.Worker, text *string) (int, bool) {
	if len(workers) == 0 {
		return 0, false
	}
	idx := rand.Intn(len(workers))
	return idx, true
}

// Name returns the policy name
func (p *RandomPolicy) Name() string {
	return "random"
}
