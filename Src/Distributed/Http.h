#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace Arvo::Distributed::Http
{
using Bytes = std::vector<std::uint8_t>;
using Socket = std::intptr_t;
inline constexpr Socket InvalidSocket = -1;

struct Message
{
    std::string FirstLine;
    std::map<std::string, std::string> Headers;
    Bytes Body;
};

struct Response
{
    int Status = 0;
    Bytes Body;
};
using CancellationProbe = std::function<bool()>;
using HeaderHandler = std::function<void(const Message&, std::size_t, int& bodyTimeoutSeconds)>;

void Initialize();
void Close(Socket socket);
Socket Listen(const std::string& address, std::uint16_t port);
Socket Accept(Socket listener);
Message Read(Socket socket, std::size_t maximumBody = 16 * 1024 * 1024,
             const std::atomic<bool>* cancelled = nullptr, int timeoutSeconds = 600);
// Called before body allocation/receipt, allowing authentication and admission
// of a large native job without retaining unauthenticated full-frame payloads.
Message ReadRequest(Socket socket, std::size_t maximumBody, const HeaderHandler&,
                    const CancellationProbe& cancelled = {}, int timeoutSeconds = 600);
void Reply(Socket socket, int status, const Bytes& body, const std::string& type = "text/plain",
           const CancellationProbe& cancelled = {});
void Reply(Socket socket, int status, const std::string& body);
Response Request(const std::string& endpoint, const std::string& method,
                 const std::string& path, const std::string& body,
                 const std::string& token = {}, int timeoutSeconds = 600,
                 const std::atomic<bool>* cancelled = nullptr,
                 std::size_t maximumResponseBytes = 16 * 1024 * 1024);
Response RequestBinary(const std::string& endpoint, const std::string& method, const std::string& path,
                       const Bytes& body, const std::string& token, int timeoutSeconds,
                       const CancellationProbe& cancelled, std::size_t maximumResponseBytes,
                       const std::string& jobId = {});
std::string Text(const Bytes& bytes);
std::map<std::string, std::string> ParseFields(const std::string& text);
std::string EncodeFields(const std::map<std::string, std::string>& fields);
} // namespace Arvo::Distributed::Http
