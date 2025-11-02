package policy

import (
	"sync/atomic"

	"github.com/sglang/sglang-router-go/internal/core"
)

// RoundRobinPolicy implements round-robin worker selection
// Similar to Rust RoundRobinPolicy
type RoundRobinPolicy struct {
	counter uint64
}

// NewRoundRobinPolicy creates a new round-robin policy
func NewRoundRobinPolicy() *RoundRobinPolicy {
	return &RoundRobinPolicy{
		counter: 0,
	}
}

// SelectWorker selects the next worker in rotation
func (p *RoundRobinPolicy) SelectWorker(workers []core.Worker, text *string) (int, bool) {
	if len(workers) == 0 {
		return 0, false
	}
	idx := atomic.AddUint64(&p.counter, 1) - 1
	return int(idx % uint64(len(workers))), true
}

// Name returns the policy name
func (p *RoundRobinPolicy) Name() string {
	return "round_robin"
}
