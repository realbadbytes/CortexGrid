#include "cortex/crypto.hpp"
#include "cortex/executor.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::string_view kProtocol = "cg-executor-v1";

std::vector<std::string> split_tabs(const std::string& line) {
  std::vector<std::string> fields;
  std::size_t start = 0;
  while (true) {
    const std::size_t end = line.find('\t', start);
    fields.push_back(line.substr(start, end - start));
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return fields;
}

std::string find_argument(int argc, char** argv, std::string_view name) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) {
      return argv[index + 1];
    }
  }
  return {};
}

std::string escape_field(std::string value) {
  for (char& character : value) {
    if (character == '\t' || character == '\n' || character == '\r') {
      character = ' ';
    }
  }
  return value;
}

int serve(const std::string& node_id) {
  const char* secret_value = std::getenv("CORTEX_NODE_PRIVATE_KEY");
  if (secret_value == nullptr) {
    std::cerr << "CORTEX_NODE_PRIVATE_KEY is required\n";
    return 2;
  }

  std::vector<unsigned char> secret;
  try {
    secret = cortex::hex_to_bytes(secret_value);
  } catch (const std::exception& error) {
    std::cerr << "Invalid CORTEX_NODE_PRIVATE_KEY: " << error.what() << '\n';
    return 2;
  }
  if (secret.size() != 32U) {
    std::cerr << "CORTEX_NODE_PRIVATE_KEY must contain 32 bytes\n";
    return 2;
  }
  const std::string public_key = cortex::ed25519_public_key(secret);

#ifdef CORTEX_HAS_CUDA
  constexpr std::string_view capability = "cuda,cpu-fallback";
#else
  constexpr std::string_view capability = "cpu";
#endif
  std::cout << "READY\t" << kProtocol << '\t' << capability << '\t'
            << public_key << '\n'
            << std::flush;

  std::string line;
  while (std::getline(std::cin, line)) {
    const auto fields = split_tabs(line);
    if (fields.size() == 1U && fields[0] == "STOP") {
      std::cout << "STOPPED\n" << std::flush;
      return 0;
    }
    if (fields.size() == 12U && fields[0] == "ATTEST") {
      try {
        cortex::ExecutionRequest request{
            .request_id = fields[1],
            .model_id = fields[2],
            .model_version = fields[3],
            .shard_index = std::stoi(fields[4]),
            .node_id = node_id,
            .input_digest = fields[5],
            .weights_digest = fields[7],
            .compute_units = std::stoi(fields[8]),
            .challenge = fields[9],
        };
        cortex::ExecutionResult result{
            .ok = true,
            .output_digest = fields[6],
            .backend = fields[10],
            .kernel_micros =
                static_cast<std::uint64_t>(std::stoull(fields[11])),
            .proof = {},
            .error = {},
        };
        result.proof =
            cortex::ed25519_sign(secret, cortex::canonical_result(request, result));
        std::cout << "ATTESTED\t" << request.request_id << '\t'
                  << result.proof << '\n'
                  << std::flush;
      } catch (const std::exception& error) {
        std::cout << "ERROR\t" << escape_field(error.what()) << '\n'
                  << std::flush;
      }
      continue;
    }
    if (fields.size() != 9U || fields[0] != "EXEC") {
      std::cout << "ERROR\tinvalid request frame\n" << std::flush;
      continue;
    }

    try {
      cortex::ExecutionRequest request{
          .request_id = fields[1],
          .model_id = fields[2],
          .model_version = fields[3],
          .shard_index = std::stoi(fields[4]),
          .node_id = node_id,
          .input_digest = fields[5],
          .weights_digest = fields[6],
          .compute_units = std::stoi(fields[7]),
          .challenge = fields[8],
      };
      const auto result = cortex::execute(request, secret);
      if (!result.ok) {
        std::cout << "ERROR\t" << escape_field(result.error) << '\n'
                  << std::flush;
        continue;
      }
      std::cout << "RESULT\t" << request.request_id << '\t'
                << result.output_digest << '\t' << result.backend << '\t'
                << result.kernel_micros << '\t' << result.proof << '\n'
                << std::flush;
    } catch (const std::exception& error) {
      std::cout << "ERROR\t" << escape_field(error.what()) << '\n'
                << std::flush;
    }
  }
  return 0;
}

int self_test() {
  const std::vector<unsigned char> secret(32U, 0x42U);
  cortex::ExecutionRequest request{
      .request_id = "self-test",
      .model_id = "cortex-test",
      .model_version = "1",
      .shard_index = 0,
      .node_id = "local",
      .input_digest = cortex::sha256("input"),
      .weights_digest = cortex::sha256("weights"),
      .compute_units = 2,
      .challenge = "challenge",
  };
  const auto result = cortex::execute(request, secret);
  if (!result.ok || result.output_digest.size() != 64U ||
      result.proof.size() != 128U) {
    std::cerr << "self-test failed: " << result.error << '\n';
    return 1;
  }
  std::cout << "ok " << result.backend << ' ' << result.output_digest << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
    return self_test();
  }
  const std::string node_id = find_argument(argc, argv, "--node-id");
  if (node_id.empty()) {
    std::cerr << "Usage: cortex_worker --serve --node-id NODE_ID\n";
    return 2;
  }
  return serve(node_id);
}
