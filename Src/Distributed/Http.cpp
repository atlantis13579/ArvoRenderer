#include "Distributed/Http.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace Arvo::Distributed::Http
{
namespace
{
#ifdef _WIN32
using NativeSocket = SOCKET;
#else
using NativeSocket = int;
#endif
NativeSocket Native(Socket value) { return static_cast<NativeSocket>(value); }

struct SocketGuard
{
    Socket Value = InvalidSocket;
    ~SocketGuard() { Close(Value); }
};

void Timeout(Socket socket, int seconds)
{
#ifdef _WIN32
    const DWORD value = static_cast<DWORD>(seconds * 1000);
#else
    const timeval value{seconds, 0};
#endif
    setsockopt(Native(socket), SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&value), sizeof(value));
    setsockopt(Native(socket), SOL_SOCKET, SO_SNDTIMEO,
               reinterpret_cast<const char*>(&value), sizeof(value));
#ifdef SO_NOSIGPIPE
    const int enabled = 1;
    setsockopt(Native(socket), SOL_SOCKET, SO_NOSIGPIPE,
               reinterpret_cast<const char*>(&enabled), sizeof(enabled));
#endif
}

void Send(Socket socket, const void* data, std::size_t length, const CancellationProbe& cancelled = {},
          int timeoutSeconds = 600)
{
    // A writable socket may still block on a multi-gigabyte send. Keep each
    // operation nonblocking so cancellation also covers upload and readback.
    struct RestoreBlocking
    {
        Socket Value;
#ifndef _WIN32
        int Flags;
#endif
        ~RestoreBlocking()
        {
#ifdef _WIN32
            u_long mode = 0;
            ioctlsocket(Native(Value), FIONBIO, &mode);
#else
            fcntl(Native(Value), F_SETFL, Flags);
#endif
        }
    };
#ifdef _WIN32
    u_long mode = 1;
    if (ioctlsocket(Native(socket), FIONBIO, &mode)) throw std::runtime_error("Cannot configure HTTP upload");
    RestoreBlocking restore{socket};
#else
    const int previousFlags = fcntl(Native(socket), F_GETFL, 0);
    if (previousFlags < 0 || fcntl(Native(socket), F_SETFL, previousFlags | O_NONBLOCK) < 0)
        throw std::runtime_error("Cannot configure HTTP upload");
    RestoreBlocking restore{socket, previousFlags};
#endif
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSeconds);
    constexpr size_t SendChunkBytes = 64 * 1024;
    constexpr long PollMicroseconds = 200000;
    auto cursor = static_cast<const char*>(data);
    while (length)
    {
        if (cancelled && cancelled()) throw std::runtime_error("HTTP request cancelled");
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("HTTP send timed out");
        fd_set writable;
        FD_ZERO(&writable); FD_SET(Native(socket), &writable);
        timeval delay{0, PollMicroseconds};
        const int ready = select(static_cast<int>(Native(socket) + 1), nullptr, &writable, nullptr, &delay);
#ifndef _WIN32
        if (ready < 0 && errno == EINTR) continue;
#endif
        if (ready < 0) throw std::runtime_error("HTTP connection failed while sending");
        if (!ready) continue;
        const int count = static_cast<int>(std::min<std::size_t>(length, SendChunkBytes));
#ifdef MSG_NOSIGNAL
        const int flags = MSG_NOSIGNAL;
#else
        const int flags = 0;
#endif
        const int sent = static_cast<int>(send(Native(socket), cursor, count, flags));
#ifndef _WIN32
        if (sent < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
#else
        if (sent < 0 && WSAGetLastError() == WSAEWOULDBLOCK) continue;
#endif
        if (sent <= 0) throw std::runtime_error("HTTP connection closed while sending");
        cursor += sent;
        length -= static_cast<std::size_t>(sent);
    }
}

struct Endpoint
{
    std::string Host, Port = "80", Prefix;
};

Endpoint ParseEndpoint(const std::string& text)
{
    if (text.rfind("http://", 0) != 0)
        throw std::runtime_error("Worker URLs must use http:// (use a trusted network or TLS reverse proxy tunnel)");
    if (text.find_first_of("\r\n\t ") != std::string::npos)
        throw std::runtime_error("Invalid worker URL");
    Endpoint result;
    const auto slash = text.find('/', 7);
    const auto authority = text.substr(7, slash == std::string::npos ? slash : slash - 7);
    if (slash != std::string::npos) result.Prefix = text.substr(slash);
    while (!result.Prefix.empty() && result.Prefix.back() == '/') result.Prefix.pop_back();
    if (!authority.empty() && authority[0] == '[')
    {
        const auto end = authority.find(']');
        if (end == std::string::npos) throw std::runtime_error("Invalid IPv6 worker URL");
        result.Host = authority.substr(1, end - 1);
        if (end + 1 < authority.size())
        {
            if (authority[end + 1] != ':') throw std::runtime_error("Invalid worker port");
            result.Port = authority.substr(end + 2);
        }
    }
    else
    {
        const auto colon = authority.find(':');
        result.Host = authority.substr(0, colon);
        if (colon != std::string::npos) result.Port = authority.substr(colon + 1);
    }
    if (result.Host.empty() || result.Port.empty() || authority.find('@') != std::string::npos)
        throw std::runtime_error("Invalid worker authority");
    return result;
}

Socket Connect(const Endpoint& endpoint, int timeoutSeconds, const CancellationProbe& cancelled)
{
    Initialize();
    addrinfo hints{};
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    addrinfo* addresses = nullptr;
    if (getaddrinfo(endpoint.Host.c_str(), endpoint.Port.c_str(), &hints, &addresses))
        throw std::runtime_error("Cannot resolve worker: " + endpoint.Host);
    Socket connected = InvalidSocket;
    for (auto* address = addresses; address; address = address->ai_next)
    {
        SocketGuard candidate{static_cast<Socket>(socket(address->ai_family, address->ai_socktype, address->ai_protocol))};
        if (candidate.Value == InvalidSocket) continue;
#ifdef _WIN32
        u_long nonblocking = 1;
        ioctlsocket(Native(candidate.Value), FIONBIO, &nonblocking);
#else
        const int originalFlags = fcntl(Native(candidate.Value), F_GETFL, 0);
        if (originalFlags < 0 || fcntl(Native(candidate.Value), F_SETFL, originalFlags | O_NONBLOCK) < 0) continue;
#endif
        const int result = connect(Native(candidate.Value), address->ai_addr, static_cast<int>(address->ai_addrlen));
        bool ready = result == 0;
        if (!ready)
        {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(std::min(3, timeoutSeconds));
            while (std::chrono::steady_clock::now() < deadline && !(cancelled && cancelled()))
            {
                fd_set writable, errors;
                FD_ZERO(&writable); FD_ZERO(&errors);
                FD_SET(Native(candidate.Value), &writable); FD_SET(Native(candidate.Value), &errors);
                timeval delay{0, 200000};
                const int selected = select(static_cast<int>(Native(candidate.Value) + 1), nullptr, &writable, &errors, &delay);
#ifndef _WIN32
                if (selected < 0 && errno == EINTR) continue;
#endif
                if (selected < 0 || (selected > 0 && FD_ISSET(Native(candidate.Value), &errors))) break;
                if (selected > 0 && FD_ISSET(Native(candidate.Value), &writable))
                {
                    int error = 0;
#ifdef _WIN32
                    int size = sizeof(error);
#else
                    socklen_t size = sizeof(error);
#endif
                    ready = getsockopt(Native(candidate.Value), SOL_SOCKET, SO_ERROR,
                                       reinterpret_cast<char*>(&error), &size) == 0 && error == 0;
                    break;
                }
            }
        }
#ifdef _WIN32
        nonblocking = 0;
        ioctlsocket(Native(candidate.Value), FIONBIO, &nonblocking);
#else
        fcntl(Native(candidate.Value), F_SETFL, originalFlags);
#endif
        if (ready)
        {
            connected = candidate.Value;
            candidate.Value = InvalidSocket;
            Timeout(connected, timeoutSeconds);
            break;
        }
    }
    freeaddrinfo(addresses);
    if (connected == InvalidSocket) throw std::runtime_error("Cannot connect to worker " + endpoint.Host + ":" + endpoint.Port);
    return connected;
}
} // namespace

void Initialize()
{
#ifdef _WIN32
    static std::once_flag once;
    std::call_once(once, [] {
        WSADATA data{};
        if (WSAStartup(MAKEWORD(2, 2), &data)) throw std::runtime_error("WSAStartup failed");
    });
#endif
}

void Close(Socket socket)
{
    if (socket == InvalidSocket) return;
#ifdef _WIN32
    closesocket(Native(socket));
#else
    close(Native(socket));
#endif
}

Socket Listen(const std::string& address, std::uint16_t port)
{
    Initialize();
    addrinfo hints{};
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    hints.ai_flags = AI_PASSIVE;
    addrinfo* addresses = nullptr;
    if (getaddrinfo(address.c_str(), std::to_string(port).c_str(), &hints, &addresses))
        throw std::runtime_error("Invalid worker bind address");
    Socket listener = InvalidSocket;
    for (auto* item = addresses; item; item = item->ai_next)
    {
        SocketGuard candidate{static_cast<Socket>(socket(item->ai_family, item->ai_socktype, item->ai_protocol))};
        if (candidate.Value == InvalidSocket) continue;
        const int enabled = 1;
#ifdef _WIN32
        setsockopt(Native(candidate.Value), SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                   reinterpret_cast<const char*>(&enabled), sizeof(enabled));
#else
        setsockopt(Native(candidate.Value), SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&enabled), sizeof(enabled));
#endif
        if (bind(Native(candidate.Value), item->ai_addr, static_cast<int>(item->ai_addrlen)) == 0 &&
            listen(Native(candidate.Value), 16) == 0)
        {
            listener = candidate.Value;
            candidate.Value = InvalidSocket;
            break;
        }
    }
    freeaddrinfo(addresses);
    if (listener == InvalidSocket) throw std::runtime_error("Cannot bind worker port " + std::to_string(port));
    return listener;
}

Socket Accept(Socket listener)
{
    const Socket socket = static_cast<Socket>(accept(Native(listener), nullptr, nullptr));
    if (socket != InvalidSocket) Timeout(socket, 30);
    return socket;
}

Message ReadRequest(Socket socket, std::size_t maximumBody, const HeaderHandler& beforeBody,
                    const CancellationProbe& cancelled, int timeoutSeconds)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSeconds);
    const auto receive = [&](char* destination, int size) {
        for (;;)
        {
            if (cancelled && cancelled()) throw std::runtime_error("HTTP request cancelled");
            if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("HTTP request timed out");
            fd_set readable;
            FD_ZERO(&readable); FD_SET(Native(socket), &readable);
            timeval delay{0, 200000};
            const int ready = select(static_cast<int>(Native(socket) + 1), &readable, nullptr, nullptr, &delay);
#ifndef _WIN32
            if (ready < 0 && errno == EINTR) continue;
#endif
            if (ready < 0) throw std::runtime_error("HTTP connection failed");
            if (ready) break;
        }
        return static_cast<int>(recv(Native(socket), destination, size, 0));
    };
    std::string pending;
    char buffer[8192];
    std::size_t headerEnd = std::string::npos;
    while ((headerEnd = pending.find("\r\n\r\n")) == std::string::npos)
    {
        const int count = receive(buffer, sizeof(buffer));
        if (count <= 0) throw std::runtime_error("HTTP response timed out or closed");
        pending.append(buffer, count);
        if (pending.size() > 32768 && pending.find("\r\n\r\n") == std::string::npos)
            throw std::runtime_error("HTTP headers exceed limit");
    }
    Message result;
    std::istringstream headers(pending.substr(0, headerEnd));
    std::getline(headers, result.FirstLine);
    if (!result.FirstLine.empty() && result.FirstLine.back() == '\r') result.FirstLine.pop_back();
    std::string line;
    while (std::getline(headers, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto colon = line.find(':');
        if (colon == std::string::npos) throw std::runtime_error("Malformed HTTP header");
        std::string name = line.substr(0, colon), value = line.substr(colon + 1);
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        while (!value.empty() && value.front() == ' ') value.erase(value.begin());
        if (!result.Headers.emplace(name, value).second) throw std::runtime_error("Duplicate HTTP header");
    }
    if (result.Headers.count("transfer-encoding")) throw std::runtime_error("Chunked transfer is not supported by the worker protocol");
    std::size_t length = 0;
    if (const auto found = result.Headers.find("content-length"); found != result.Headers.end())
    {
        std::size_t used = 0;
        const auto parsed = std::stoull(found->second, &used);
        if (used != found->second.size() || parsed > maximumBody) throw std::runtime_error("HTTP body exceeds limit");
        length = static_cast<std::size_t>(parsed);
    }
    const auto first = pending.begin() + static_cast<std::ptrdiff_t>(headerEnd + 4);
    if (size_t(pending.end() - first) > length) throw std::runtime_error("Unexpected HTTP body bytes");
    int bodyTimeout = timeoutSeconds;
    if (beforeBody) beforeBody(result, length, bodyTimeout);
    if (bodyTimeout <= 0) throw std::runtime_error("Invalid HTTP body timeout");
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(bodyTimeout);
    if (cancelled && cancelled()) throw std::runtime_error("HTTP request cancelled");
    result.Body.reserve(length);
    result.Body.assign(first, pending.end());
    while (result.Body.size() < length)
    {
        const int wanted = static_cast<int>(std::min<std::size_t>(sizeof(buffer), length - result.Body.size()));
        const int count = receive(buffer, wanted);
        if (count <= 0) throw std::runtime_error("Truncated HTTP response");
        result.Body.insert(result.Body.end(), buffer, buffer + count);
    }
    return result;
}
Message Read(Socket socket, std::size_t maximumBody, const std::atomic<bool>* cancelled, int timeoutSeconds)
{
    return ReadRequest(socket, maximumBody, {}, [=] { return cancelled && cancelled->load(); }, timeoutSeconds);
}

void Reply(Socket socket, int status, const Bytes& body, const std::string& type,
           const CancellationProbe& cancelled)
{
    const std::string header = "HTTP/1.1 " + std::to_string(status) + " Arvo\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\nContent-Type: " + type + "\r\nConnection: close\r\n\r\n";
    Send(socket, header.data(), header.size(), cancelled);
    Send(socket, body.data(), body.size(), cancelled);
}

void Reply(Socket socket, int status, const std::string& body)
{
    Reply(socket, status, Bytes(body.begin(), body.end()));
}

namespace
{
Response RequestData(const std::string& endpoint, const std::string& method,
                     const std::string& path, const void* data, size_t bytes, const char* contentType,
                     const std::string& token, int timeoutSeconds, const CancellationProbe& cancelled,
                     std::size_t maximumResponseBytes, const std::string& jobId)
{
    const Endpoint parsed = ParseEndpoint(endpoint);
    if (token.find_first_of("\r\n") != std::string::npos) throw std::runtime_error("Invalid bearer token");
    if (jobId.size() > 128 || jobId.find_first_of("\r\n") != std::string::npos)
        throw std::runtime_error("Invalid HTTP job identifier");
    // Connect already polls cancellation; use the same probe for all I/O stages.
    SocketGuard socket{Connect(parsed, timeoutSeconds, cancelled)};
    const std::string host = parsed.Host.find(':') == std::string::npos ? parsed.Host : "[" + parsed.Host + "]";
    const std::string header = method + " " + parsed.Prefix + path + " HTTP/1.1\r\nHost: " + host + ":" + parsed.Port +
        "\r\nContent-Length: " + std::to_string(bytes) + "\r\nContent-Type: " + contentType + "\r\nConnection: close\r\n" +
        (jobId.empty() ? std::string{} : "X-Arvo-Job: " + jobId + "\r\n") +
        (token.empty() ? std::string{} : "Authorization: Bearer " + token + "\r\n") + "\r\n";
    Send(socket.Value, header.data(), header.size(), cancelled, timeoutSeconds);
    Send(socket.Value, data, bytes, cancelled, timeoutSeconds);
    auto message = ReadRequest(socket.Value, maximumResponseBytes, {}, cancelled, timeoutSeconds);
    std::istringstream status(message.FirstLine);
    std::string version;
    Response result;
    status >> version >> result.Status;
    if (version != "HTTP/1.1" && version != "HTTP/1.0") throw std::runtime_error("Invalid HTTP response");
    result.Body = std::move(message.Body);
    return result;
}
} // namespace
Response Request(const std::string& endpoint, const std::string& method, const std::string& path,
                 const std::string& body, const std::string& token, int timeoutSeconds,
                 const std::atomic<bool>* cancelled, std::size_t maximumResponseBytes)
{
    return RequestData(endpoint, method, path, body.data(), body.size(), "text/plain", token,
        timeoutSeconds, [=] { return cancelled && cancelled->load(); }, maximumResponseBytes, {});
}
Response RequestBinary(const std::string& endpoint, const std::string& method, const std::string& path,
                       const Bytes& body, const std::string& token, int timeoutSeconds,
                       const CancellationProbe& cancelled, std::size_t maximumResponseBytes,
                       const std::string& jobId)
{
    return RequestData(endpoint, method, path, body.data(), body.size(), "application/octet-stream", token,
                       timeoutSeconds, cancelled, maximumResponseBytes, jobId);
}

std::string Text(const Bytes& bytes) { return {bytes.begin(), bytes.end()}; }

std::map<std::string, std::string> ParseFields(const std::string& text)
{
    std::map<std::string, std::string> fields;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.empty()) continue;
        const auto equals = line.find('=');
        if (equals == std::string::npos || equals == 0 || !fields.emplace(line.substr(0, equals), line.substr(equals + 1)).second)
            throw std::runtime_error("Malformed or duplicate protocol field");
    }
    return fields;
}

std::string EncodeFields(const std::map<std::string, std::string>& fields)
{
    std::string result;
    for (const auto& [key, value] : fields)
    {
        if (key.find_first_of("\n\r=") != std::string::npos || value.find_first_of("\n\r") != std::string::npos)
            throw std::runtime_error("Invalid protocol field");
        result += key + "=" + value + "\n";
    }
    return result;
}
} // namespace Arvo::Distributed::Http
