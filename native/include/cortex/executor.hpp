#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cortex {

inline constexpr std::size_t kActivationWords = 16'384;

struct ExecutionRequest {
  std::string request_id;
  std::string model_id;
  std::string model_version;
  int shard_index = 0;
  std::string node_id;
  std::string input_digest;
  std::string weights_digest;
  int compute_units = 0;
  std::string challenge;
};

struct ExecutionResult {
  bool ok = false;
  std::string output_digest;
  std::string backend;
  std::uint64_t kernel_micros = 0;
  std::string proof;
  std::string error;
};

std::string canonical_result(const ExecutionRequest& request,
                             const ExecutionResult& result);
ExecutionResult execute(const ExecutionRequest& request,
                        const std::vector<unsigned char>& node_secret);
void run_cpu_activation(std::vector<std::uint32_t>& activation,
                        std::uint32_t shard_seed,
                        int rounds);

#ifdef CORTEX_HAS_CUDA
bool run_cuda_activation(std::vector<std::uint32_t>& activation,
                         std::uint32_t shard_seed,
                         int rounds,
                         std::string& error);
#endif

}  // namespace cortex
