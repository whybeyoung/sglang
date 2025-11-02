package router

import (
	"context"
	"net/http"

	"github.com/sglang/sglang-router-go/internal/protocols"
)

// Router is the interface for routing requests
// Similar to Rust RouterTrait
type Router interface {
	// RouteChat routes a chat completion request
	RouteChat(ctx context.Context, request *protocols.ChatCompletionRequest, modelID *string) (interface{}, error)

	// RouteChatStream routes a streaming chat completion request
	RouteChatStream(ctx context.Context, request *protocols.ChatCompletionRequest, modelID *string, w http.ResponseWriter) error

	// RouteGenerate routes a generate request
	RouteGenerate(ctx context.Context, request *protocols.GenerateRequest, modelID *string) (interface{}, error)

	// RouteGenerateStream routes a streaming generate request
	RouteGenerateStream(ctx context.Context, request *protocols.GenerateRequest, modelID *string, w http.ResponseWriter) error
}
