#include "Distributed/LocalWorkers.h"
#include "Distributed/Http.h"
#include "Distributed/Protocol.h"
#include "Core/Paths.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#ifndef _WIN32
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace Arvo::Distributed
{
namespace
{
constexpr auto StartupTimeout = std::chrono::seconds(20);
constexpr auto StartupPoll = std::chrono::milliseconds(50);
constexpr int ProbeTimeoutSeconds = 1;
constexpr uint32_t ChildExitTimeoutMilliseconds = 2000;
struct Child
{
#ifdef _WIN32
    HANDLE Process = nullptr, Job = nullptr;
    ~Child()
    {
        if (Job) CloseHandle(Job);
        if (Process) { WaitForSingleObject(Process, ChildExitTimeoutMilliseconds); CloseHandle(Process); }
    }
    bool Running() const { return Process && WaitForSingleObject(Process, 0) == WAIT_TIMEOUT; }
    static std::wstring Quote(const std::wstring& value)
    {
        std::wstring result = L"\"";
        size_t slashes = 0;
        for (const auto ch : value)
        {
            if (ch == L'\\') { ++slashes; continue; }
            result.append(slashes * (ch == L'\"' ? 2 : 1), L'\\');
            if (ch == L'\"') result += L'\\';
            result += ch;
            slashes = 0;
        }
        result.append(slashes * 2, L'\\');
        return result + L'\"';
    }
    void Start(const std::filesystem::path& executable, const std::filesystem::path& content,
               uint16_t port, const std::string& token, const std::filesystem::path& log)
    {
        Job = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!Job || !SetInformationJobObject(Job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)))
            throw std::runtime_error("Cannot create local worker lifetime group");
        const auto wideToken = std::filesystem::u8path(token).wstring();
        auto command = Quote(executable.wstring()) + L" --bind 127.0.0.1 --port " + std::to_wstring(port) +
            L" --backend cpu --content-root " + Quote(content.wstring()) + L" --token " + Quote(wideToken);
        SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        const HANDLE output = CreateFileW(log.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                          &inherit, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        const HANDLE input = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                         &inherit, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (output == INVALID_HANDLE_VALUE || input == INVALID_HANDLE_VALUE)
        {
            if (output != INVALID_HANDLE_VALUE) CloseHandle(output);
            if (input != INVALID_HANDLE_VALUE) CloseHandle(input);
            throw std::runtime_error("Cannot open local worker log");
        }
        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = input; startup.hStdOutput = startup.hStdError = output;
        PROCESS_INFORMATION process{};
        const bool created = CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE,
            CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, executable.parent_path().c_str(), &startup, &process);
        CloseHandle(output); CloseHandle(input);
        if (!created) throw std::runtime_error("Cannot start ArvoWorker beside ArvoEditor");
        Process = process.hProcess;
        if (!AssignProcessToJobObject(Job, Process) || ResumeThread(process.hThread) == DWORD(-1))
        {
            TerminateProcess(Process, 1);
            CloseHandle(process.hThread);
            throw std::runtime_error("Cannot attach local worker to editor lifetime");
        }
        CloseHandle(process.hThread);
    }
#else
    pid_t Pid = -1;
    ~Child() { if (Pid > 0) { kill(Pid, SIGTERM); while (waitpid(Pid, nullptr, 0) < 0 && errno == EINTR) {} } }
    bool Running()
    {
        if (Pid <= 0) return false;
        if (waitpid(Pid, nullptr, WNOHANG) == 0) return true;
        Pid = -1;
        return false;
    }
    void Start(const std::filesystem::path& executable, const std::filesystem::path& content,
               uint16_t port, const std::string& token, const std::filesystem::path& log)
    {
        std::vector<std::string> args{executable.string(), "--bind", "127.0.0.1", "--port", std::to_string(port),
            "--backend", "cpu", "--content-root", content.string(), "--token", token};
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        argv.push_back(nullptr);
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
        posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
        const int error = posix_spawn(&Pid, executable.c_str(), &actions, nullptr, argv.data(), environ);
        posix_spawn_file_actions_destroy(&actions);
        if (error) { Pid = -1; throw std::runtime_error("Cannot start ArvoWorker beside ArvoEditor"); }
    }
#endif
};
}
struct LocalWorkers::Impl
{
    struct Entry
    {
        uint16_t Port;
        std::string Url;
        std::unique_ptr<Child> Owned;
    };
    std::filesystem::path Content, Executable, Logs;
    std::string Token;
    std::vector<Entry> Entries;
    std::atomic<bool> Stopping{false};
    std::mutex StartupMutex;
    mutable std::mutex StatusMutex;
    std::string Message = "Starting two local workers...";
    std::future<void> Startup;
    ~Impl() { Stopping = true; if (Startup.valid()) Startup.wait(); }
    void SetStatus(std::string text) { std::lock_guard<std::mutex> lock(StatusMutex); Message = std::move(text); }
    bool Probe(const std::string& url)
    {
        try
        {
            const auto response = Http::Request(url, "GET", "/status", "", Token, ProbeTimeoutSeconds, &Stopping);
            if (response.Status != 200) return false;
            const auto fields = Http::ParseFields(Http::Text(response.Body));
            return fields.at("version") == Protocol::Version && fields.count("backend") && fields.count("status");
        }
        catch (...) { return false; }
    }
    void Ensure(const std::vector<std::string>& servers, const std::atomic<bool>* cancelled)
    {
        std::lock_guard<std::mutex> lock(StartupMutex);
        const auto stop = [&] { return Stopping || (cancelled && cancelled->load()); };
        try
        {
            for (auto& entry : Entries)
            {
                if (std::find(servers.begin(), servers.end(), entry.Url) == servers.end()) continue;
                if (stop()) throw std::runtime_error("Local worker startup cancelled");
                if (Probe(entry.Url)) continue;
                if (!entry.Owned || !entry.Owned->Running())
                {
                    // A failed handshake is not permission to replace another process.
                    Http::Socket freePort;
                    try { freePort = Http::Listen("127.0.0.1", entry.Port); }
                    catch (...) { throw std::runtime_error(entry.Url + " is occupied or requires a different access token"); }
                    Http::Close(freePort);
                    if (!std::filesystem::is_regular_file(Executable))
                        throw std::runtime_error("ArvoWorker executable is missing beside ArvoEditor");
                    std::filesystem::create_directories(Logs);
                    entry.Owned = std::make_unique<Child>();
                    entry.Owned->Start(Executable, Content, entry.Port, Token,
                                      Logs / ("worker-" + std::to_string(entry.Port) + ".log"));
                }
                const auto deadline = std::chrono::steady_clock::now() + StartupTimeout;
                while (!Probe(entry.Url))
                {
                    if (stop()) throw std::runtime_error("Local worker startup cancelled");
                    if (!entry.Owned->Running() || std::chrono::steady_clock::now() >= deadline)
                        throw std::runtime_error("Local worker failed to start; see " + Logs.string());
                    std::this_thread::sleep_for(StartupPoll);
                }
            }
            SetStatus("Local workers ready");
        }
        catch (const std::exception& error) { SetStatus(error.what()); throw; }
    }
};
LocalWorkers::LocalWorkers(std::filesystem::path contentRoot, std::string token,
                          std::array<uint16_t, 2> ports, std::filesystem::path executable) : Impl_(std::make_unique<Impl>())
{
    Impl_->Content = std::filesystem::absolute(contentRoot);
    Impl_->Token = std::move(token);
    Impl_->Executable = executable.empty() ? Core::ExecutableDirectory() /
#ifdef _WIN32
        "ArvoWorker.exe"
#else
        "ArvoWorker"
#endif
        : std::filesystem::absolute(executable);
    Impl_->Logs = Core::BinaryRoot() / "Logs";
    std::vector<std::string> servers;
    for (const auto port : ports)
    {
        servers.push_back("http://127.0.0.1:" + std::to_string(port));
        Impl_->Entries.push_back({port, servers.back(), {}});
    }
    Impl_->Startup = std::async(std::launch::async, [state = Impl_.get(), servers] {
        try { state->Ensure(servers, nullptr); } catch (...) { /* Status retains the actionable error. */ }
    });
}
LocalWorkers::~LocalWorkers() = default;
void LocalWorkers::EnsureReady(const std::vector<std::string>& servers, const std::atomic<bool>* cancelled)
{ Impl_->Ensure(servers, cancelled); }
std::string LocalWorkers::Status() const
{ std::lock_guard<std::mutex> lock(Impl_->StatusMutex); return Impl_->Message; }
}
