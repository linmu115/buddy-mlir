#pragma once

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <ifaddrs.h>
#include <memory>
#include <mpi.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>

namespace buddy::runtime {

enum class TpTransport { Mpi, Tcp, TcpSpin };

inline TpTransport parseTpTransport(const std::string &value) {
  if (value == "mpi")
    return TpTransport::Mpi;
  if (value == "tcp")
    return TpTransport::Tcp;
  if (value == "tcp-spin")
    return TpTransport::TcpSpin;
  throw std::invalid_argument("TP transport expects mpi, tcp or tcp-spin");
}

// A single ordered stream. The worker command determines the payload length;
// both stages enter the same sequence of fixed-size reductions for that frame.
class TpSocket {
  static int checked(int rc, const char *what) {
    if (rc < 0)
      throw std::system_error(errno, std::generic_category(), what);
    return rc;
  }

  struct Fd {
    int value = -1;
    ~Fd() {
      if (value >= 0)
        close(value);
    }
    Fd() = default;
    Fd(const Fd &) = delete;
    Fd &operator=(const Fd &) = delete;
  };
  Fd stream;
  bool spin;

  static in_addr localAddress(std::string iface) {
    if (iface.empty())
      for (const char *key :
           {"BUDDY_TP_TCP_IFACE", "MPIR_CVAR_NETWORK_IFACE"}) {
        const char *v = std::getenv(key);
        if (v && *v) {
          iface = v;
          break;
        }
      }
    if (iface.empty())
      return {htonl(INADDR_LOOPBACK)};
    ifaddrs *head = nullptr;
    checked(getifaddrs(&head), "TP getifaddrs");
    std::unique_ptr<ifaddrs, decltype(&freeifaddrs)> guard(head, freeifaddrs);
    for (auto *it = head; it; it = it->ifa_next)
      if (it->ifa_addr && it->ifa_addr->sa_family == AF_INET &&
          iface == it->ifa_name)
        return reinterpret_cast<sockaddr_in *>(it->ifa_addr)->sin_addr;
    throw std::invalid_argument("TP interface has no IPv4 address: " + iface);
  }
  static void option(int fd, int level, int key, int value) {
    checked(setsockopt(fd, level, key, &value, sizeof(value)), "TP setsockopt");
  }
  static void buffers(int fd) {
    option(fd, SOL_SOCKET, SO_SNDBUF, 1 << 20);
    option(fd, SOL_SOCKET, SO_RCVBUF, 1 << 20);
  }
  void quickAck() const {
#ifdef TCP_QUICKACK
    int on = 1;
    setsockopt(stream.value, IPPROTO_TCP, TCP_QUICKACK, &on, sizeof(on));
#endif
  }
  static size_t transferred(ssize_t n, const char *what) {
    if (n > 0)
      return size_t(n);
    if (n == 0)
      throw std::runtime_error("TP peer closed during transfer");
    if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
      return 0;
    throw std::system_error(errno, std::generic_category(), what);
  }

public:
  TpSocket(MPI_Comm comm, int rank, bool spin, const std::string &iface)
      : spin(spin) {
    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_addr = localAddress(iface);
    Fd listener;
    int &fd = rank == 0 ? listener.value : stream.value;
    fd = checked(socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0), "TP socket");
    buffers(fd); // Set before handshake for negotiated window scaling.
    if (!rank)
      option(fd, SOL_SOCKET, SO_REUSEADDR, 1);
    checked(bind(fd, reinterpret_cast<sockaddr *>(&local), sizeof(local)),
            "TP bind");
    if (!rank) {
      checked(listen(fd, 1), "TP listen");
      socklen_t len = sizeof(local);
      checked(getsockname(fd, reinterpret_cast<sockaddr *>(&local), &len),
              "TP getsockname");
    }
    // Both address and port are already in network byte order.
    struct Endpoint {
      in_addr address;
      in_port_t port;
    } endpoint{local.sin_addr, local.sin_port}, other{};
    if (MPI_Sendrecv(&endpoint, sizeof(endpoint), MPI_BYTE, 1 - rank, 534,
                     &other, sizeof(other), MPI_BYTE, 1 - rank, 534, comm,
                     MPI_STATUS_IGNORE) != MPI_SUCCESS)
      throw std::runtime_error("TP endpoint exchange failed");
    if (!rank) {
      do {
        stream.value = accept4(fd, nullptr, nullptr, SOCK_CLOEXEC);
      } while (stream.value < 0 && errno == EINTR);
      checked(stream.value, "TP accept");
    } else {
      sockaddr_in server{};
      server.sin_family = AF_INET;
      server.sin_addr = other.address;
      server.sin_port = other.port;
      checked(
          connect(fd, reinterpret_cast<sockaddr *>(&server), sizeof(server)),
          "TP connect");
    }
    option(stream.value, IPPROTO_TCP, TCP_NODELAY, 1);
    quickAck();
    if (spin) {
      int flags = checked(fcntl(stream.value, F_GETFL), "TP F_GETFL");
      checked(fcntl(stream.value, F_SETFL, flags | O_NONBLOCK),
              "TP O_NONBLOCK");
#ifdef SO_BUSY_POLL
      int usec = 50;
      setsockopt(stream.value, SOL_SOCKET, SO_BUSY_POLL, &usec, sizeof(usec));
#endif
    }
  }

  void transfer(const void *out, size_t outBytes, void *in, size_t inBytes) {
    using Clock = std::chrono::steady_clock;
    auto idleSince = Clock::time_point{};
    auto *src = static_cast<const char *>(out);
    auto *dst = static_cast<char *>(in);
    while (outBytes || inBytes) {
      size_t sent = 0, received = 0;
      if (outBytes) {
        sent = transferred(
            send(stream.value, src, outBytes, MSG_DONTWAIT | MSG_NOSIGNAL),
            "TP send");
        src += sent;
        outBytes -= sent;
      }
      if (inBytes) {
        received = transferred(recv(stream.value, dst, inBytes, MSG_DONTWAIT),
                               "TP recv");
        dst += received;
        inBytes -= received;
      }
      if (sent || received) {
        idleSince = {};
        continue;
      }
      if (spin) {
        auto now = Clock::now();
        if (idleSince == Clock::time_point{})
          idleSince = now;
        if (now - idleSince < std::chrono::milliseconds(20))
          continue;
      }
      short events = (outBytes ? POLLOUT : 0) | (inBytes ? POLLIN : 0);
      pollfd pfd{stream.value, events, 0};
      int rc = poll(&pfd, 1, spin ? 20 : -1);
      if (rc < 0 && errno == EINTR)
        continue;
      checked(rc, "TP poll");
      if ((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) &&
          !(pfd.revents & events))
        throw std::runtime_error("TP socket disconnected");
    }
  }
};
} // namespace buddy::runtime
