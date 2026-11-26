// A rank-zero worker keeps the BPE ranks and compiled Jinja template resident.
// Use a socketpair rather than a shell so request text is always data.
#pragma once
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <spawn.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;

namespace buddy::runtime {
class ChatTokenizer {
  pid_t pid = -1;
  int fd = -1;
  FILE *input = nullptr;

public:
  ChatTokenizer(const ChatTokenizer &) = delete;
  ChatTokenizer &operator=(const ChatTokenizer &) = delete;
  explicit ChatTokenizer(const std::filesystem::path &directory) {
    int sockets[2];
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets))
      throw std::runtime_error("tokenizer socketpair failed");
    auto script = (directory / "qwen_bpe.py").string();
    auto dir = directory.string();
    char python[] = "python3";
    char *argv[] = {python, script.data(), dir.data(), nullptr};
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, sockets[1], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, sockets[1], STDOUT_FILENO);
    int error = posix_spawnp(&pid, python, &actions, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(sockets[1]);
    if (error) {
      close(sockets[0]);
      throw std::runtime_error("could not launch python3 tokenizer");
    }
    fd = sockets[0];
    input = fdopen(fd, "r");
    if (!input) {
      close(fd);
      waitpid(pid, nullptr, 0);
      throw std::runtime_error("could not open tokenizer stream");
    }
  }
  ~ChatTokenizer() {
    fclose(input);
    while (waitpid(pid, nullptr, 0) == -1 && errno == EINTR) {
    }
  }

  std::vector<int64_t> encode(const nlohmann::json &request) {
    std::string line = request.dump() + "\n";
    for (size_t pos = 0; pos < line.size();) {
      auto n = send(fd, line.data() + pos, line.size() - pos, MSG_NOSIGNAL);
      if (n < 0 && errno == EINTR)
        continue;
      if (n <= 0)
        throw std::runtime_error("tokenizer worker disconnected");
      pos += n;
    }
    char *buffer = nullptr;
    size_t capacity = 0;
    auto n = getline(&buffer, &capacity, input);
    std::unique_ptr<char, decltype(&free)> owned(buffer, free);
    if (n <= 0)
      throw std::runtime_error(
          "tokenizer worker failed; check python3 and Jinja2");
    auto response = nlohmann::json::parse(buffer, buffer + n);
    if (response.contains("error"))
      throw std::invalid_argument(response["error"].get<std::string>());
    return response.at("ids").get<std::vector<int64_t>>();
  }
};
} // namespace buddy::runtime
