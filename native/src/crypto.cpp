#include "cortex/crypto.hpp"

#include <openssl/evp.h>
#include <openssl/sha.h>

#include <array>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace cortex {
namespace {

std::string to_hex(std::span<const unsigned char> bytes) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (const unsigned char value : bytes) {
    stream << std::setw(2) << static_cast<int>(value);
  }
  return stream.str();
}

unsigned char hex_nibble(char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<unsigned char>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<unsigned char>(value - 'a' + 10);
  }
  if (value >= 'A' && value <= 'F') {
    return static_cast<unsigned char>(value - 'A' + 10);
  }
  throw std::invalid_argument("invalid hexadecimal value");
}

}  // namespace

std::string sha256(std::string_view input) {
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
  SHA256(reinterpret_cast<const unsigned char*>(input.data()), input.size(),
         digest.data());
  return to_hex(digest);
}

std::string sha256_bytes(std::span<const std::uint32_t> input) {
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
  SHA256(reinterpret_cast<const unsigned char*>(input.data()),
         input.size_bytes(), digest.data());
  return to_hex(digest);
}

std::string ed25519_sign(std::span<const unsigned char> private_key,
                         std::string_view input) {
  EVP_PKEY* key = EVP_PKEY_new_raw_private_key(
      EVP_PKEY_ED25519, nullptr, private_key.data(), private_key.size());
  if (key == nullptr) {
    throw std::runtime_error("unable to load Ed25519 private key");
  }
  EVP_MD_CTX* context = EVP_MD_CTX_new();
  if (context == nullptr) {
    EVP_PKEY_free(key);
    throw std::runtime_error("unable to allocate signature context");
  }

  std::array<unsigned char, 64> signature{};
  std::size_t signature_length = signature.size();
  const bool initialized =
      EVP_DigestSignInit(context, nullptr, nullptr, nullptr, key) == 1;
  const bool signed_message =
      initialized &&
      EVP_DigestSign(
          context, signature.data(), &signature_length,
          reinterpret_cast<const unsigned char*>(input.data()), input.size()) ==
          1;
  EVP_MD_CTX_free(context);
  EVP_PKEY_free(key);
  if (!signed_message) {
    throw std::runtime_error("unable to sign execution receipt");
  }
  return to_hex(
      std::span<const unsigned char>(signature.data(), signature_length));
}

std::string ed25519_public_key(
    std::span<const unsigned char> private_key) {
  EVP_PKEY* key = EVP_PKEY_new_raw_private_key(
      EVP_PKEY_ED25519, nullptr, private_key.data(), private_key.size());
  if (key == nullptr) {
    throw std::runtime_error("unable to load Ed25519 private key");
  }
  std::array<unsigned char, 32> public_key{};
  std::size_t public_key_length = public_key.size();
  const bool exported =
      EVP_PKEY_get_raw_public_key(key, public_key.data(), &public_key_length) ==
      1;
  EVP_PKEY_free(key);
  if (!exported) {
    throw std::runtime_error("unable to derive Ed25519 public key");
  }
  return to_hex(
      std::span<const unsigned char>(public_key.data(), public_key_length));
}

std::vector<unsigned char> hex_to_bytes(std::string_view input) {
  if (input.size() % 2 != 0) {
    throw std::invalid_argument("hex input must have an even length");
  }
  std::vector<unsigned char> output;
  output.reserve(input.size() / 2);
  for (std::size_t index = 0; index < input.size(); index += 2) {
    output.push_back(static_cast<unsigned char>(
        (hex_nibble(input[index]) << 4U) | hex_nibble(input[index + 1])));
  }
  return output;
}

}  // namespace cortex
