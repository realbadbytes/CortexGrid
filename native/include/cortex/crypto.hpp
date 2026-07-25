#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace cortex {

std::string sha256(std::string_view input);
std::string sha256_bytes(std::span<const std::uint32_t> input);
std::string ed25519_sign(std::span<const unsigned char> private_key,
                         std::string_view input);
std::string ed25519_public_key(std::span<const unsigned char> private_key);
std::vector<unsigned char> hex_to_bytes(std::string_view input);

}  // namespace cortex
