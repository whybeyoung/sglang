// Package config provides configuration management
package config

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config represents the application configuration
type Config struct {
	Tokenizer struct {
		Path string `json:"path"`
	} `json:"tokenizer"`
	GRPC struct {
		Endpoint string `json:"endpoint"`
		Timeout  int    `json:"timeout"` // seconds
	} `json:"grpc"`
	Generation struct {
		Temperature       float32 `json:"temperature"`
		TopP              float32 `json:"top_p"`
		TopK              int32   `json:"top_k"`
		MaxNewTokens      int32   `json:"max_new_tokens"`
		SkipSpecialTokens bool    `json:"skip_special_tokens"`
		Stream            bool    `json:"stream"`
	} `json:"generation"`
}

// LoadFromFile loads configuration from a JSON file
func LoadFromFile(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	return &config, nil
}

// LoadFromEnv loads configuration from environment variables
func LoadFromEnv() *Config {
	config := &Config{}

	if path := os.Getenv("SGL_TOKENIZER_PATH"); path != "" {
		config.Tokenizer.Path = path
	}

	if endpoint := os.Getenv("SGL_GRPC_ENDPOINT"); endpoint != "" {
		config.GRPC.Endpoint = endpoint
	}

	return config
}

// DefaultConfig returns a default configuration
func DefaultConfig() *Config {
	return &Config{
		Tokenizer: struct {
			Path string `json:"path"`
		}{
			Path: "/path/to/tokenizer", // Directory path (recommended) or tokenizer.json file path
		},
		GRPC: struct {
			Endpoint string `json:"endpoint"`
			Timeout  int    `json:"timeout"`
		}{
			Endpoint: "grpc://localhost:20000",
			Timeout:  30,
		},
		Generation: struct {
			Temperature       float32 `json:"temperature"`
			TopP              float32 `json:"top_p"`
			TopK              int32   `json:"top_k"`
			MaxNewTokens      int32   `json:"max_new_tokens"`
			SkipSpecialTokens bool    `json:"skip_special_tokens"`
			Stream            bool    `json:"stream"`
		}{
			Temperature:       1.0,
			TopP:              1.0,
			TopK:              -1,
			MaxNewTokens:      100,
			SkipSpecialTokens: true,
			Stream:            false,
		},
	}
}
