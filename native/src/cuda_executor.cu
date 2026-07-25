#include "cortex/executor.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace cortex {
namespace {

__device__ std::uint32_t rotate_left_device(std::uint32_t value,
                                            unsigned int amount) {
  const unsigned int shift = amount & 31U;
  return shift == 0U
             ? value
             : static_cast<std::uint32_t>((value << shift) |
                                          (value >> (32U - shift)));
}

__global__ void activation_round(const std::uint32_t* current,
                                 std::uint32_t* next,
                                 std::size_t size,
                                 std::uint32_t round_constant,
                                 int round) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }
  const std::uint32_t neighbor = current[(index + 1U) % size];
  next[index] =
      rotate_left_device(current[index] ^ round_constant,
                         static_cast<unsigned int>((index + round) % 31U + 1U)) +
      neighbor + static_cast<std::uint32_t>(index);
}

std::string cuda_message(cudaError_t status) {
  return std::string(cudaGetErrorName(status)) + ": " +
         cudaGetErrorString(status);
}

}  // namespace

bool run_cuda_activation(std::vector<std::uint32_t>& activation,
                         std::uint32_t shard_seed,
                         int rounds,
                         std::string& error) {
  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    error = status == cudaSuccess ? "no CUDA device available"
                                  : cuda_message(status);
    return false;
  }

  const std::size_t bytes = activation.size() * sizeof(std::uint32_t);
  std::uint32_t* current = nullptr;
  std::uint32_t* next = nullptr;
  status = cudaMalloc(&current, bytes);
  if (status == cudaSuccess) {
    status = cudaMalloc(&next, bytes);
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(current, activation.data(), bytes,
                        cudaMemcpyHostToDevice);
  }

  constexpr int threads = 256;
  const int blocks =
      static_cast<int>((activation.size() + threads - 1U) / threads);
  for (int round = 0; status == cudaSuccess && round < rounds; ++round) {
    const std::uint32_t round_constant =
        shard_seed + static_cast<std::uint32_t>(round) * 0x9e3779b9U;
    activation_round<<<blocks, threads>>>(current, next, activation.size(),
                                          round_constant, round);
    status = cudaGetLastError();
    std::swap(current, next);
  }
  if (status == cudaSuccess) {
    status =
        cudaMemcpy(activation.data(), current, bytes, cudaMemcpyDeviceToHost);
  }

  if (current != nullptr) {
    cudaFree(current);
  }
  if (next != nullptr) {
    cudaFree(next);
  }
  if (status != cudaSuccess) {
    error = cuda_message(status);
    return false;
  }
  return true;
}

}  // namespace cortex
