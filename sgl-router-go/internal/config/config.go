package config

import (
	"flag"
	"fmt"
	"strings"

	"github.com/sglang/sglang-router-go/internal/core"
)

// Config represents the router configuration
// Similar to Rust RouterConfig
type Config struct {
	Host       string
	Port       uint16
	WorkerURLs []string
	Policy     string

	// Tokenizer configuration
	TokenizerPath *string
	ModelPath     *string

	// Connection mode
	ConnectionMode core.ConnectionMode

	// gRPC specific
	GRPCEnabled bool

	// Logging
	LogLevel string
	LogDir   *string

	// Metrics
	MetricsEnabled bool
	MetricsHost    string
	MetricsPort    uint16
}

// LoadFromFlags loads configuration from command line flags
func LoadFromFlags() (*Config, error) {
	cfg := &Config{}

	flag.StringVar(&cfg.Host, "host", "0.0.0.0", "Host to bind to")
	var port uint
	flag.UintVar(&port, "port", 30000, "Port to bind to")
	cfg.Port = uint16(port)

	var workerURLs string
	flag.StringVar(&workerURLs, "worker-urls", "", "Comma-separated list of worker URLs (grpc://host:port)")

	flag.StringVar(&cfg.Policy, "policy", "cache_aware", "Load balancing policy (random, round_robin, cache_aware, power_of_two)")

	var tokenizerPath string
	flag.StringVar(&tokenizerPath, "tokenizer-path", "", "Path to tokenizer.json file")
	if tokenizerPath != "" {
		cfg.TokenizerPath = &tokenizerPath
	}

	var modelPath string
	flag.StringVar(&modelPath, "model-path", "", "Path to model (HuggingFace ID or local path)")
	if modelPath != "" {
		cfg.ModelPath = &modelPath
	}

	flag.StringVar(&cfg.LogLevel, "log-level", "info", "Log level (debug, info, warn, error)")

	var logDir string
	flag.StringVar(&logDir, "log-dir", "", "Log directory (optional)")
	if logDir != "" {
		cfg.LogDir = &logDir
	}

	flag.BoolVar(&cfg.MetricsEnabled, "enable-metrics", false, "Enable Prometheus metrics")
	flag.StringVar(&cfg.MetricsHost, "metrics-host", "0.0.0.0", "Metrics host")
	var metricsPort uint
	flag.UintVar(&metricsPort, "metrics-port", 29000, "Metrics port")
	cfg.MetricsPort = uint16(metricsPort)

	flag.Parse()

	// Parse worker URLs
	if workerURLs != "" {
		cfg.WorkerURLs = strings.Split(workerURLs, ",")
		// Trim whitespace
		for i, url := range cfg.WorkerURLs {
			cfg.WorkerURLs[i] = strings.TrimSpace(url)
		}
	}

	// Determine connection mode from worker URLs
	if len(cfg.WorkerURLs) > 0 {
		firstURL := cfg.WorkerURLs[0]
		if strings.HasPrefix(firstURL, "grpc://") {
			cfg.ConnectionMode = core.ConnectionModeGRPC
			cfg.GRPCEnabled = true
		} else {
			cfg.ConnectionMode = core.ConnectionModeHTTP
		}
	} else {
		cfg.ConnectionMode = core.ConnectionModeGRPC // Default to gRPC
		cfg.GRPCEnabled = true
	}

	// Validate configuration
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return cfg, nil
}

// Validate validates the configuration
func (c *Config) Validate() error {
	// Validate policy
	validPolicies := map[string]bool{
		"random":       true,
		"round_robin":  true,
		"cache_aware":  true,
		"power_of_two": true,
	}
	if !validPolicies[c.Policy] {
		return fmt.Errorf("invalid policy: %s", c.Policy)
	}

	// For gRPC mode, tokenizer is required
	if c.GRPCEnabled && c.TokenizerPath == nil && c.ModelPath == nil {
		return fmt.Errorf("tokenizer-path or model-path is required for gRPC mode")
	}

	return nil
}
