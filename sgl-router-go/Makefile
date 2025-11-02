.PHONY: generate build test clean proto

# Protobuf generation
PROTO_DIR = proto
PROTO_OUT = pkg/proto
INTERNAL_PROTO_DIR = internal/grpc/proto

generate: proto

proto:
	@echo "Generating protobuf code..."
	@mkdir -p $(PROTO_OUT) $(INTERNAL_PROTO_DIR)
	protoc --go_out=$(PROTO_OUT) --go_opt=paths=source_relative \
		--go-grpc_out=$(PROTO_OUT) --go-grpc_opt=paths=source_relative \
		--proto_path=$(PROTO_DIR) \
		$(PROTO_DIR)/*.proto
	protoc --go_out=$(INTERNAL_PROTO_DIR) --go_opt=paths=source_relative \
		--go-grpc_out=$(INTERNAL_PROTO_DIR) --go-grpc_opt=paths=source_relative \
		--proto_path=../sgl-router/src/proto \
		../sgl-router/src/proto/sglang_scheduler.proto

build:
	@echo "Building sglang-router..."
	@mkdir -p bin
	go build -o bin/sglang-router ./cmd/router

test:
	go test ./...

clean:
	rm -rf bin $(PROTO_OUT) $(INTERNAL_PROTO_DIR)

run:
	./bin/sglang-router $(ARGS)
