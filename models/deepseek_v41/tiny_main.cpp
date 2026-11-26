// Run exported tiny weights without loading any reference outputs.
#include "TinyBackbone.h"
#include <iostream>
int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: tiny-deepseek-v41 WEIGHT_DIR DECODE_STEPS TOKEN...\n";
    return 2;
  }
  using namespace buddy::v4;
  using namespace buddy::v4::tiny;
  try {
    Weights w(argv[1], "weights");
    Ids prompt;
    for (int i = 3; i < argc; i++)
      prompt.push_back(std::stoi(argv[i]));
    auto tokens = Runner(w).run(prompt, std::stoi(argv[2]));
    for (auto t : tokens)
      std::cout << t << ' ';
    std::cout << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
