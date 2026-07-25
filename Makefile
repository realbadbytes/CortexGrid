.PHONY: build build-model-runtime build-provider-runtime bootstrap-model bootstrap-provider run run-real run-5tb join term node test test-real check clean

BUILD_DIR ?= build
CMAKE_BUILD_TYPE ?= Release
PORT ?= 4173
LLAMA_DIR ?= third_party/llama.cpp
LLAMA_BUILD_DIR ?= $(BUILD_DIR)/llama
MODEL_FILE ?= models/SmolLM2-135M-Instruct-Q4_K_M.gguf
LLAMA_COMMIT ?= 720d7fa4097f76e5d0eade5a92c1df87c1faf9d9
MODEL_URL ?= https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf
LLAMA_CUDA ?= $(shell command -v nvcc >/dev/null 2>&1 && echo ON || echo OFF)
COORDINATOR_URL ?= http://127.0.0.1:4173
COORDINATOR_PUBLIC_KEY ?=
NODE_ID ?= provider-one
NODE_NAME ?= $(NODE_ID)
OPERATOR ?= Independent
REGION ?= unknown
MEMORY_GIB ?= 24
ADVERTISE_HOST ?= 127.0.0.1
PROVIDER_API_PORT ?= 51800
PROVIDER_RPC_PORT ?= 50052
PROVIDER_PRICE_MICRO ?= 10

build:
	cmake -S native -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	cmake --build $(BUILD_DIR) --parallel

build-model-runtime:
	test -f $(LLAMA_DIR)/CMakeLists.txt
	cmake -S $(LLAMA_DIR) -B $(LLAMA_BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DGGML_RPC=ON -DGGML_CUDA=$(LLAMA_CUDA) -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF
	cmake --build $(LLAMA_BUILD_DIR) --parallel 4 --target ggml-rpc-server llama-server

build-provider-runtime:
	test -f $(LLAMA_DIR)/CMakeLists.txt
	cmake -S $(LLAMA_DIR) -B $(LLAMA_BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DGGML_RPC=ON -DGGML_CUDA=$(LLAMA_CUDA) -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF
	cmake --build $(LLAMA_BUILD_DIR) --parallel 4 --target ggml-rpc-server

bootstrap-model:
	mkdir -p third_party models
	test -d $(LLAMA_DIR)/.git || git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git $(LLAMA_DIR)
	git -C $(LLAMA_DIR) fetch --depth 1 origin $(LLAMA_COMMIT)
	git -C $(LLAMA_DIR) checkout --detach FETCH_HEAD
	$(MAKE) build-model-runtime
	test -f $(MODEL_FILE) || curl -L --fail --max-time 600 --output $(MODEL_FILE) $(MODEL_URL)
	sha256sum $(MODEL_FILE)

bootstrap-provider:
	mkdir -p third_party
	test -d $(LLAMA_DIR)/.git || git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git $(LLAMA_DIR)
	git -C $(LLAMA_DIR) fetch --depth 1 origin $(LLAMA_COMMIT)
	git -C $(LLAMA_DIR) checkout --detach FETCH_HEAD
	$(MAKE) build
	$(MAKE) build-provider-runtime

run: build
	PYTHONPATH=control CORTEX_EXECUTOR_PATH=$(CURDIR)/$(BUILD_DIR)/cortex_worker PORT=$(PORT) python3 -m cortexgrid.server

run-real: build build-model-runtime
	test -f $(MODEL_FILE)
	PYTHONPATH=control CORTEX_REAL_MODEL=on CORTEX_EXECUTOR_PATH=$(CURDIR)/$(BUILD_DIR)/cortex_worker CORTEX_MODEL_PATH=$(CURDIR)/$(MODEL_FILE) CORTEX_RPC_BINARY=$(CURDIR)/$(LLAMA_BUILD_DIR)/bin/ggml-rpc-server CORTEX_MODEL_SERVER_BINARY=$(CURDIR)/$(LLAMA_BUILD_DIR)/bin/llama-server PORT=$(PORT) python3 -m cortexgrid.server

run-5tb: build
	PYTHONPATH=control CORTEX_MODEL_PROFILE=5tb CORTEX_EXECUTOR_PATH=$(CURDIR)/$(BUILD_DIR)/cortex_worker PORT=$(PORT) python3 -m cortexgrid.server

join: build build-provider-runtime
	PYTHONPATH=control python3 -m cortexgrid.node_daemon --coordinator $(COORDINATOR_URL) $(if $(COORDINATOR_PUBLIC_KEY),--coordinator-public-key $(COORDINATOR_PUBLIC_KEY),) --node-id $(NODE_ID) --name "$(NODE_NAME)" --operator "$(OPERATOR)" --region $(REGION) --memory-gib $(MEMORY_GIB) --price-micro-per-work-unit $(PROVIDER_PRICE_MICRO) --advertise-host $(ADVERTISE_HOST) --api-port $(PROVIDER_API_PORT) --rpc-port $(PROVIDER_RPC_PORT) --executor $(CURDIR)/$(BUILD_DIR)/cortex_worker --rpc-binary $(CURDIR)/$(LLAMA_BUILD_DIR)/bin/ggml-rpc-server

term:
	PYTHONPATH=control python3 -m cortexgrid.terminal --server $(COORDINATOR_URL)

node:
	PYTHONPATH=control python3 -m cortexgrid.provider_cli status --node-id $(NODE_ID) --api http://127.0.0.1:$(PROVIDER_API_PORT)

test: build
	PYTHONPATH=control CORTEX_EXECUTOR_PATH=$(CURDIR)/$(BUILD_DIR)/cortex_worker python3 -m unittest discover -s tests -v

test-real: build build-model-runtime
	test -f $(MODEL_FILE)
	CORTEX_TEST_REAL_MODEL=1 PYTHONPATH=control CORTEX_EXECUTOR_PATH=$(CURDIR)/$(BUILD_DIR)/cortex_worker CORTEX_MODEL_PATH=$(CURDIR)/$(MODEL_FILE) CORTEX_RPC_BINARY=$(CURDIR)/$(LLAMA_BUILD_DIR)/bin/ggml-rpc-server CORTEX_MODEL_SERVER_BINARY=$(CURDIR)/$(LLAMA_BUILD_DIR)/bin/llama-server python3 -m unittest tests/test_real_model.py -v

check:
	python3 -m compileall -q control
	cmake -S native -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	cmake --build $(BUILD_DIR) --parallel

clean:
	cmake --build $(BUILD_DIR) --target clean
