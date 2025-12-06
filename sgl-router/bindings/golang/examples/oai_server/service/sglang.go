package service

import (
	"os"

	sglang "github.com/sglang/sglang-go-grpc-sdk"
)

// SGLangService wraps SGLang client
type SGLangService struct {
	client *sglang.Client
}

// NewSGLangService creates a new SGLang service
func NewSGLangService(endpoint, tokenizerPath string) (*SGLangService, error) {
	// Check if gRPC client mode is enabled via environment variable
	useGrpcClient := os.Getenv("USE_GRPC_CLIENT") == "true"

	client, err := sglang.NewClient(sglang.ClientConfig{
		Endpoint:      endpoint,
		TokenizerPath: tokenizerPath,
		UseGrpcClient: useGrpcClient, // Enable optimized gRPC client if requested
	})
	if err != nil {
		return nil, err
	}

	return &SGLangService{
		client: client,
	}, nil
}

// Client returns the underlying SGLang client
func (s *SGLangService) Client() *sglang.Client {
	return s.client
}

// Close closes the SGLang client
func (s *SGLangService) Close() error {
	if s.client != nil {
		return s.client.Close()
	}
	return nil
}
