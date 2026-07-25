#include "cortex/executor.hpp"

#include "cortex/crypto.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace cortex {
namespace {

std::uint32_t rotate_left(std::uint32_t value, unsigned int amount) {
  const unsigned int shift = amount & 31U;
  if (shift == 0U) {
    return value;
  }
  return static_cast<std::uint32_t>((value << shift) |
                                    (value >> (32U - shift)));
}

std::uint32_t parse_seed(std::string_view digest, std::size_t offset) {
  if (digest.size() < offset + 8U) {
    throw std::invalid_argument("digest is too short");
  }
  const auto bytes = hex_to_bytes(digest.substr(offset, 8));
  return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
         (static_cast<std::uint32_t>(bytes[1]) << 16U) |
         (static_cast<std::uint32_t>(bytes[2]) << 8U) |
         static_cast<std::uint32_t>(bytes[3]);
}

std::vector<std::uint32_t> initialize_activation(
    const ExecutionRequest& request) {
  const std::string seed_digest =
      sha256(request.input_digest + "|" + request.weights_digest);
  std::uint32_t state = parse_seed(seed_digest, 0);
  const std::uint32_t mix = parse_seed(seed_digest, 8);
  std::vector<std::uint32_t> activation(kActivationWords);

  for (std::size_t index = 0; index < activation.size(); ++index) {
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    activation[index] =
        state ^ mix ^ static_cast<std::uint32_t>(index * 0x9e3779b9ULL);
  }
  return activation;
}

}  // namespace

void run_cpu_activation(std::vector<std::uint32_t>& activation,
                        std::uint32_t shard_seed,
                        int rounds) {
  std::vector<std::uint32_t> next(activation.size());
  for (int round = 0; round < rounds; ++round) {
    const std::uint32_t round_constant =
        shard_seed + static_cast<std::uint32_t>(round) * 0x9e3779b9U;
    for (std::size_t index = 0; index < activation.size(); ++index) {
      const std::uint32_t current = activation[index];
      const std::uint32_t neighbor =
          activation[(index + 1U) % activation.size()];
      next[index] =
          rotate_left(current ^ round_constant,
                      static_cast<unsigned int>((index + round) % 31U + 1U)) +
          neighbor + static_cast<std::uint32_t>(index);
    }
    activation.swap(next);
  }
}

std::string canonical_result(const ExecutionRequest& request,
                             const ExecutionResult& result) {
  return request.request_id + "|" + request.model_id + "|" +
         request.model_version + "|" + std::to_string(request.shard_index) +
         "|" + request.node_id + "|" + request.input_digest + "|" +
         result.output_digest + "|" + request.weights_digest + "|" +
         std::to_string(request.compute_units) + "|" + request.challenge +
         "|" + result.backend + "|" + std::to_string(result.kernel_micros);
}

ExecutionResult execute(const ExecutionRequest& request,
                        const std::vector<unsigned char>& node_secret) {
  ExecutionResult result;
  if (request.compute_units <= 0 || request.compute_units > 10'000) {
    result.error = "compute units are outside the supported range";
    return result;
  }
  if (request.input_digest.size() != 64U ||
      request.weights_digest.size() != 64U) {
    result.error = "activation and weight digests must be SHA-256 values";
    return result;
  }

  try {
    auto activation = initialize_activation(request);
    const std::uint32_t shard_seed =
        parse_seed(request.weights_digest, 0) ^
        static_cast<std::uint32_t>(request.shard_index);
    const auto started = std::chrono::steady_clock::now();

#ifdef CORTEX_HAS_CUDA
    std::string cuda_error;
    if (run_cuda_activation(activation, shard_seed, request.compute_units,
                            cuda_error)) {
      result.backend = "cuda";
    } else {
      run_cpu_activation(activation, shard_seed, request.compute_units);
      result.backend = "cpu-fallback";
    }
#else
    run_cpu_activation(activation, shard_seed, request.compute_units);
    result.backend = "cpu";
#endif

    const auto finished = std::chrono::steady_clock::now();
    result.kernel_micros =
        static_cast<std::uint64_t>(std::chrono::duration_cast<
                                      std::chrono::microseconds>(finished -
                                                                 started)
                                      .count());
    result.output_digest = sha256_bytes(activation);
    result.ok = true;
    result.proof =
        ed25519_sign(node_secret, canonical_result(request, result));
    return result;
  } catch (const std::exception& error) {
    result.error = error.what();
    return result;
  }
}

}  // namespace cortex
