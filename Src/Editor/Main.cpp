#include "Core/Paths.h"
#include "Editor/ModuleAdapters.h"
#include "Editor/EditorPresenter.h"
#include "Editor/RenderPanel.h"
#include "Editor/Imgui.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <windowsx.h>
#include <shellapi.h>

namespace
{
using Arvo::Editor::EditorCore;
using Arvo::Editor::Image;
using Arvo::Editor::Rect;
constexpr wchar_t WindowClassName[] = L"ArvoStudioWindow";
constexpr int EditorPanelX = 10;
constexpr int EditorPanelY = 10;
constexpr int EditorPanelWidth = 330;
constexpr int EditorPanelGap = 10;
constexpr int EditorViewportMargin = 10;
std::unique_ptr<EditorCore> CreateEditor()
{
    auto editor = std::make_unique<EditorCore>(Arvo::Core::ContentRoot());
    Arvo::Editor::RegisterModuleAdapters(*editor);
    editor->SetPickHandler(Arvo::Editor::CreatePickHandler());
    editor->RequestRender();
    return editor;
}
struct WindowState
{
    std::unique_ptr<EditorCore> Editor;
    Arvo::Editor::RenderPanel Panel;
    std::unique_ptr<Arvo::Editor::IEditorPresenter> Presenter;
    Arvo::Editor::EditorPresentationAPI LastRealtimePresentation =
        Arvo::Editor::EditorPresentationAPI::DX11;
    std::string PresentationSelection;
    std::string PendingPresentationSelection;
    std::string FailedPresentationSelection;
    std::string PresentationError;
    void* NativeWindow = nullptr;
    std::chrono::steady_clock::time_point LastTick = std::chrono::steady_clock::now();
    int ClientWidth = 1280;
    int ClientHeight = 720;
    int MouseX = 0;
    int MouseY = 0;
    int MouseWheel = 0;
    unsigned char MouseButtons = 0;
    bool CameraRotateCapture = false;
    std::ostream* PresentationTrace = nullptr;
    std::uint64_t PresentedFrames = 0;
    std::uint64_t PresentationFailures = 0;
    bool DrawingFrame = false;
    bool ClosePending = false;
    bool Closing = false;
    bool ResizePending = false;
    std::filesystem::path SmokeDirectory;
    int SmokeStage = 0;
    bool SmokeRender = false;
    int ExitCode = 0;
    std::chrono::steady_clock::time_point SmokeStarted = std::chrono::steady_clock::now();
};

void TracePresentation(WindowState& state, const std::string& stage)
{
    if (state.PresentationTrace != nullptr)
        *state.PresentationTrace << GetTickCount64() << " " << stage << std::endl;
}

bool IsRealtimeBackend(std::string_view backendId)
{
    return backendId.rfind("renderer.", 0) == 0;
}

bool EnsurePresentationBackend(WindowState& state)
{
    const std::string selectedBackend = state.Editor->SelectedBackend();
    const Arvo::Editor::EditorPresentationAPI desiredAPI =
        Arvo::Editor::ResolvePresentationAPI(
            selectedBackend,
            state.LastRealtimePresentation);
    if (selectedBackend == state.PresentationSelection &&
        state.Presenter != nullptr && state.Presenter->API() == desiredAPI)
    {
        return true;
    }
    if (selectedBackend == state.FailedPresentationSelection &&
        state.Presenter != nullptr)
    {
        return true;
    }
    if (state.Presenter != nullptr && state.Presenter->API() == desiredAPI)
    {
        state.PresentationSelection = selectedBackend;
        state.FailedPresentationSelection.clear();
        state.PresentationError.clear();
        if (IsRealtimeBackend(selectedBackend))
        {
            state.LastRealtimePresentation = desiredAPI;
        }
        return true;
    }

    // Keep the old window API alive while the new offscreen RHI initializes.
    // Concurrent DX12 teardown / Vulkan ICD initialization can fail inside the driver.
    if (state.Presenter != nullptr && state.Editor->ViewportBackend() != selectedBackend)
    {
        if (state.PendingPresentationSelection != selectedBackend)
            TracePresentation(state, "waiting for first frame: " + selectedBackend);
        state.PendingPresentationSelection = selectedBackend;
        return true;
    }
    state.PendingPresentationSelection.clear();
    TracePresentation(state, "switch begin: " + state.PresentationSelection + " -> " + selectedBackend);
    const bool hadPreviousPresenter = state.Presenter != nullptr;
    const Arvo::Editor::EditorPresentationAPI previousAPI = hadPreviousPresenter
        ? state.Presenter->API()
        : Arvo::Editor::EditorPresentationAPI::DX11;
    if (state.Presenter != nullptr)
    {
        // A Win32 window can only own one active flip-model swap chain. Release
        // it before initializing the newly selected native presentation API.
        TracePresentation(state, "previous presenter shutdown begin");
        state.Presenter->Shutdown();
        state.Presenter.reset();
        TracePresentation(state, "previous presenter shutdown complete");
    }
    TracePresentation(state, "new presenter initialize begin");

    std::unique_ptr<Arvo::Editor::IEditorPresenter> candidate =
        Arvo::Editor::CreateEditorPresenter(desiredAPI);
    if (candidate == nullptr)
    {
        state.PresentationError = std::string(Arvo::Editor::ToString(desiredAPI))
            + " editor presentation is not available on this platform.";
    }
    else if (!candidate->Initialize(
                 state.NativeWindow,
                 state.ClientWidth,
                 state.ClientHeight))
    {
        state.PresentationError = candidate->LastError();
    }
    else
    {
        TracePresentation(state, "new presenter initialize complete");
        state.Presenter = std::move(candidate);
        state.PresentationSelection = selectedBackend;
        state.FailedPresentationSelection.clear();
        state.PresentationError.clear();
        if (IsRealtimeBackend(selectedBackend))
        {
            state.LastRealtimePresentation = desiredAPI;
        }
        return true;
    }
    // Release partially initialized swapchain resources before recreating the old API.
    ++state.PresentationFailures;
    TracePresentation(state, "FAIL initialize: " + state.PresentationError);
    candidate.reset();
    state.FailedPresentationSelection = selectedBackend;

    // Keep the editor controllable after an unavailable API selection, while
    // preserving the visible error and never reporting the fallback as native.
    if (hadPreviousPresenter)
    {
        std::unique_ptr<Arvo::Editor::IEditorPresenter> previous =
            Arvo::Editor::CreateEditorPresenter(previousAPI);
        if (previous != nullptr && previous->Initialize(
                state.NativeWindow,
                state.ClientWidth,
                state.ClientHeight))
        {
            state.Presenter = std::move(previous);
        }
    }
    return state.Presenter != nullptr;
}

Rect EditorViewportBounds(const WindowState& state)
{
    const float left = static_cast<float>(EditorPanelX + EditorPanelWidth + EditorPanelGap);
    const float top = static_cast<float>(EditorViewportMargin);
    const float availableWidth = std::max(1.0f, static_cast<float>(state.ClientWidth) - left - EditorViewportMargin);
    const float availableHeight = std::max(1.0f, static_cast<float>(state.ClientHeight) - top - EditorViewportMargin);
    const float width = std::min(availableWidth, availableHeight * 16.0f / 9.0f);
    const float height = width * 9.0f / 16.0f;
    return {left + (availableWidth - width) * 0.5f, top + (availableHeight - height) * 0.5f, width, height};
}

void DrawEditorImgui(WindowState& state)
{
    state.Panel.Draw(*state.Editor, state.ClientHeight);
    if (imguiTextFocused()) state.Editor->ClearCameraInput();
}

bool DrawEditorFrame(
    WindowState& state,
    Image* capture,
    bool present,
    std::string* error = nullptr)
{
    // Native swapchain operations can synchronously dispatch window messages.
    // Never replace a presenter or begin another frame while one is recording.
    if (state.DrawingFrame)
    {
        TracePresentation(state, "deferred reentrant draw");
        if (capture != nullptr)
        {
            if (error != nullptr) *error = "Cannot capture a nested editor frame.";
            return false;
        }
        return true;
    }
    state.DrawingFrame = true;
    struct FrameGuard
    {
        WindowState& State;
        ~FrameGuard()
        {
            State.DrawingFrame = false;
            if (State.ClosePending)
            {
                State.ClosePending = false;
                PostMessageW(static_cast<HWND>(State.NativeWindow), WM_CLOSE, 0, 0);
            }
        }
    } guard{state};
    const auto resetFailedPresenter = [&state, error](const std::string& message)
    {
        ++state.PresentationFailures;
        TracePresentation(state, "FAIL draw: " + message);
        if (error != nullptr) *error = message;
        state.PresentationError = message;
        state.PresentationSelection.clear();
        if (state.Presenter != nullptr)
        {
            state.Presenter->Shutdown();
            state.Presenter.reset();
        }
        return false;
    };
    if (!EnsurePresentationBackend(state) || state.Presenter == nullptr ||
        !state.Presenter->IsInitialized())
    {
        if (error != nullptr) *error = state.PresentationError;
        return false;
    }

    if (state.ResizePending)
    {
        state.ResizePending = false;
        if (!state.Presenter->Resize(state.ClientWidth, state.ClientHeight))
            return resetFailedPresenter(state.Presenter->LastError());
    }
    const Rect viewport = EditorViewportBounds(state);
    if (!state.Presenter->BeginFrame(
            state.Editor->ViewportImage(),
            state.Editor->ViewportRevision(),
            static_cast<int>(viewport.X),
            static_cast<int>(viewport.Y),
            static_cast<int>(viewport.Width),
            static_cast<int>(viewport.Height)))
    {
        return resetFailedPresenter(state.Presenter->LastError());
    }

    const int imguiMouseY = state.ClientHeight - 1 - state.MouseY;
    imguiBeginFrame(
        state.MouseX,
        imguiMouseY,
        state.MouseButtons,
        state.MouseWheel);
    state.MouseWheel = 0;
    DrawEditorImgui(state);
    imguiEndFrame();
    state.Presenter->DrawImgui();
    if (capture != nullptr && !state.Presenter->CaptureBackBuffer(*capture))
    {
        return resetFailedPresenter(state.Presenter->LastError());
    }
    if (present && !state.Presenter->EndFrame())
    {
        return resetFailedPresenter(state.Presenter->LastError());
    }
    if (present) ++state.PresentedFrames;
    return true;
}

void RenderEditorWindow(WindowState& state)
{
    if (state.Closing || IsIconic(static_cast<HWND>(state.NativeWindow))) return;
    (void)DrawEditorFrame(state, nullptr, true);
}

void ResizeEditor(WindowState& state, int width, int height)
{
    state.ClientWidth = std::max(width, 1);
    state.ClientHeight = std::max(height, 1);
    // Resize GPU resources at the next frame boundary, including during modal sizing.
    state.ResizePending = true;

    const Rect viewport = EditorViewportBounds(state);
    state.Editor->ResizeViewport(
        static_cast<std::uint32_t>(std::max(viewport.Width, 1.0f)),
        static_cast<std::uint32_t>(std::max(viewport.Height, 1.0f)));
}

void AdvanceSmoke(WindowState& state)
{
    if (state.SmokeDirectory.empty() || state.Closing) return;
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - state.SmokeStarted).count() > 120)
    {
        state.ExitCode = 10;
        PostMessageW(static_cast<HWND>(state.NativeWindow), WM_CLOSE, 0, 0);
        return;
    }
    if (state.SmokeRender && state.SmokeStage == 2)
    {
        const auto progress = state.Panel.Progress();
        if (progress.State == Arvo::Distributed::JobState::Running)
        {
            if (progress.CompletedTiles > 0 && progress.CompletedTiles < progress.TotalTiles)
            {
                Image captured;
                if (DrawEditorFrame(state, &captured, true))
                    Arvo::Distributed::WritePng(state.SmokeDirectory / "render-progress.png", captured.Width, captured.Height, captured.PixelsRGBA8);
            }
            return;
        }
        if (progress.State != Arvo::Distributed::JobState::Completed || progress.Servers.size() != 2 ||
            progress.Servers[0].CompletedTiles == 0 || progress.Servers[1].CompletedTiles == 0)
        {
            state.ExitCode = 12;
            std::ofstream(state.SmokeDirectory / "editor-render-error.txt") << progress.Error;
        }
        else
        {
            Image captured;
            if (DrawEditorFrame(state, &captured, true))
                Arvo::Distributed::WritePng(state.SmokeDirectory / "render-completed.png", captured.Width, captured.Height, captured.PixelsRGBA8);
            std::ofstream(state.SmokeDirectory / "editor-render.txt") << "UI render complete: " << progress.OutputPath.string() << "\n"
                << progress.CompletedTiles << "/" << progress.TotalTiles << " tiles; worker tiles "
                << progress.Servers[0].CompletedTiles << "," << progress.Servers[1].CompletedTiles << "\n";
        }
        PostMessageW(static_cast<HWND>(state.NativeWindow), WM_CLOSE, 0, 0);
        return;
    }
    if (!state.Editor->ViewportImage().IsValid() ||
        state.Editor->ViewportBackend() != state.Editor->SelectedBackend()) return;
    try
    {
        Image captured;
        std::string error;
        if (!DrawEditorFrame(state, &captured, true, &error)) throw std::runtime_error(error);
        Arvo::Distributed::WritePng(state.SmokeDirectory / (state.SmokeStage == 0 ? "preview-dx11.png" : "preview-dxr.png"),
            captured.Width, captured.Height, captured.PixelsRGBA8);
        if (state.SmokeStage == 0)
        {
            const auto backends = state.Editor->Backends();
            const bool hasDxr = std::any_of(backends.begin(), backends.end(), [](const auto& b) { return b.Id == "renderer.dxr"; });
            if (hasDxr)
            {
                state.Editor->SelectBackend("renderer.dxr");
                state.Editor->SetCameraMovementKey('W', true);
                state.Editor->Tick(0.1f);
                state.Editor->SetCameraMovementKey('W', false);
                ++state.SmokeStage;
                return;
            }
        }
        if (state.PresentationFailures) throw std::runtime_error("Native presentation failed");
        std::ofstream report(state.SmokeDirectory / "editor-smoke.txt");
        report << "Native preview, ImGui and camera update passed; DXR=" << (state.SmokeStage != 0) << "\n";
        if (state.SmokeRender)
        {
            state.Panel.Start(*state.Editor);
            state.SmokeStage = 2;
            return;
        }
    }
    catch (const std::exception& e)
    {
        state.ExitCode = 11;
        std::filesystem::create_directories(state.SmokeDirectory);
        std::ofstream(state.SmokeDirectory / "editor-smoke-error.txt") << e.what();
    }
    PostMessageW(static_cast<HWND>(state.NativeWindow), WM_CLOSE, 0, 0);
}

LRESULT CALLBACK WindowProcedure(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
    auto* state = reinterpret_cast<WindowState*>(GetWindowLongPtrW(window, GWLP_USERDATA));
    if (message == WM_NCCREATE)
    {
        const auto* create = reinterpret_cast<const CREATESTRUCTW*>(lParam);
        state = static_cast<WindowState*>(create->lpCreateParams);
        SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(state));
    }
    if (state == nullptr)
    {
        return DefWindowProcW(window, message, wParam, lParam);
    }

    switch (message)
    {
    case WM_CREATE:
    {
        RECT client = {};
        GetClientRect(window, &client);
        state->ClientWidth = std::max<LONG>(client.right - client.left, 1);
        state->ClientHeight = std::max<LONG>(client.bottom - client.top, 1);
        state->NativeWindow = window;
        if (!EnsurePresentationBackend(*state))
        {
            MessageBoxA(
                window,
                state->PresentationError.c_str(),
                "Arvo Editor",
                MB_OK | MB_ICONERROR);
            return -1;
        }
        ResizeEditor(*state, state->ClientWidth, state->ClientHeight);
        SetTimer(window, 1, 16, nullptr);
        return 0;
    }
    case WM_SIZE:
        if (wParam != SIZE_MINIMIZED)
        {
            ResizeEditor(*state, LOWORD(lParam), HIWORD(lParam));
            InvalidateRect(window, nullptr, FALSE);
        }
        return 0;
    case WM_TIMER:
    {
        const auto now = std::chrono::steady_clock::now();
        const float delta = std::chrono::duration<float>(now - state->LastTick).count();
        state->LastTick = now;
        state->Editor->Tick(std::min(delta, 0.1f));
        AdvanceSmoke(*state);
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    }
    case WM_MOUSEMOVE:
        state->MouseX = GET_X_LPARAM(lParam);
        state->MouseY = GET_Y_LPARAM(lParam);
        state->Editor->HandleCameraPointer(
            static_cast<float>(state->MouseX),
            static_cast<float>(state->MouseY),
            state->CameraRotateCapture && (wParam & MK_RBUTTON) != 0);
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    case WM_LBUTTONDOWN:
    {
        state->MouseX = GET_X_LPARAM(lParam);
        state->MouseY = GET_Y_LPARAM(lParam);
        state->MouseButtons |= IMGUI_MBUT_LEFT;
        SetCapture(window);
        const Rect viewport = EditorViewportBounds(*state);
        if (viewport.Contains(
                static_cast<float>(state->MouseX),
                static_cast<float>(state->MouseY)))
        {
            state->Editor->HandleViewportPointer(
                static_cast<float>(state->MouseX),
                static_cast<float>(state->MouseY),
                viewport);
        }
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    }
    case WM_LBUTTONUP:
        state->MouseX = GET_X_LPARAM(lParam);
        state->MouseY = GET_Y_LPARAM(lParam);
        state->MouseButtons &= static_cast<unsigned char>(~IMGUI_MBUT_LEFT);
        if (state->MouseButtons == 0)
        {
            ReleaseCapture();
        }
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    case WM_RBUTTONDOWN:
        state->MouseX = GET_X_LPARAM(lParam);
        state->MouseY = GET_Y_LPARAM(lParam);
        state->MouseButtons |= IMGUI_MBUT_RIGHT;
        state->CameraRotateCapture = EditorViewportBounds(*state).Contains(
            static_cast<float>(state->MouseX),
            static_cast<float>(state->MouseY));
        state->Editor->HandleCameraPointer(
            static_cast<float>(state->MouseX),
            static_cast<float>(state->MouseY),
            state->CameraRotateCapture);
        SetCapture(window);
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    case WM_RBUTTONUP:
        state->MouseX = GET_X_LPARAM(lParam);
        state->MouseY = GET_Y_LPARAM(lParam);
        state->MouseButtons &= static_cast<unsigned char>(~IMGUI_MBUT_RIGHT);
        state->CameraRotateCapture = false;
        state->Editor->HandleCameraPointer(
            static_cast<float>(state->MouseX),
            static_cast<float>(state->MouseY),
            false);
        if (state->MouseButtons == 0)
        {
            ReleaseCapture();
        }
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    case WM_CAPTURECHANGED:
        state->MouseButtons = 0;
        state->CameraRotateCapture = false;
        state->Editor->HandleCameraPointer(0.0f, 0.0f, false);
        return 0;
    case WM_CHAR:
        imguiTextCharacter(static_cast<unsigned int>(wParam));
        return 0;
    case WM_KEYDOWN:
        if (!imguiTextFocused()) state->Editor->SetCameraMovementKey(static_cast<char>(wParam), true);
        return 0;
    case WM_KEYUP:
        state->Editor->SetCameraMovementKey(static_cast<char>(wParam), false);
        return 0;
    case WM_KILLFOCUS:
        state->MouseButtons = 0;
        state->CameraRotateCapture = false;
        state->Editor->ClearCameraInput();
        return 0;
    case WM_MOUSEWHEEL:
    {
        POINT cursor{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
        ScreenToClient(window, &cursor);
        state->MouseX = cursor.x;
        state->MouseY = cursor.y;
        state->MouseWheel += GET_WHEEL_DELTA_WPARAM(wParam) > 0 ? -1 : 1;
        InvalidateRect(window, nullptr, FALSE);
        return 0;
    }
    case WM_ERASEBKGND:
        return 1;
    case WM_PAINT:
    {
        PAINTSTRUCT paint = {};
        BeginPaint(window, &paint);
        EndPaint(window, &paint);
        RenderEditorWindow(*state);
        return 0;
    }
    case WM_CLOSE:
        if (state->DrawingFrame)
        {
            state->ClosePending = true;
            return 0;
        }
        state->Closing = true;
        state->Panel.Cancel();
        DestroyWindow(window);
        return 0;
    case WM_DESTROY:
        state->Closing = true;
        KillTimer(window, 1);
        if (state->Presenter != nullptr)
        {
            state->Presenter->Shutdown();
            state->Presenter.reset();
        }
        PostQuitMessage(state->ExitCode);
        return 0;
    default:
        return DefWindowProcW(window, message, wParam, lParam);
    }
}

bool RegisterEditorWindowClass(HINSTANCE instance)
{
    WNDCLASSEXW windowClass = {};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    windowClass.lpfnWndProc = WindowProcedure;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    windowClass.hIcon = LoadIconW(nullptr, MAKEINTRESOURCEW(32512));
    windowClass.lpszClassName = WindowClassName;
    if (RegisterClassExW(&windowClass))
    {
        return true;
    }
    return GetLastError() == ERROR_CLASS_ALREADY_EXISTS;
}

int RunWindow(HINSTANCE instance, int showCommand, const std::filesystem::path& tracePath, const std::filesystem::path& smokeDirectory = {}, bool smokeRender = false)
{
    std::ofstream trace;
    if (!tracePath.empty())
    {
        trace.open(tracePath);
        if (!trace) return 1;
    }
    WindowState state;
    if (trace.is_open()) state.PresentationTrace = &trace;
    state.Editor = CreateEditor();
    state.SmokeDirectory = smokeDirectory;
    state.SmokeRender = smokeRender;

    if (!RegisterEditorWindowClass(instance))
    {
        return 3;
    }

    HWND window = CreateWindowExW(
        0,
        WindowClassName,
        L"Arvo Renderer Editor",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        1440,
        900,
        nullptr,
        nullptr,
        instance,
        &state);
    if (window == nullptr)
    {
        return 4;
    }
    ShowWindow(window, smokeDirectory.empty() ? showCommand : SW_SHOWNOACTIVATE);
    UpdateWindow(window);

    MSG message = {};
    while (GetMessageW(&message, nullptr, 0, 0) > 0)
    {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    return static_cast<int>(message.wParam);
}

}
int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int showCommand)
{
    try
    {
        int count = 0;
        LPWSTR* args = CommandLineToArgvW(GetCommandLineW(), &count);
        std::filesystem::path smokeDirectory;
        bool smokeRender = false;
        for (int i = 1; i < count; ++i)
        {
            if (std::wstring(args[i]) == L"--ui-smoke" && i + 1 < count) smokeDirectory = args[++i];
            else if (std::wstring(args[i]) == L"--ui-render-smoke" && i + 1 < count)
            { smokeDirectory = args[++i]; smokeRender = true; }
        }
        LocalFree(args);
        return RunWindow(instance, showCommand, {}, smokeDirectory, smokeRender);
    }
    catch (const std::exception& e)
    {
        MessageBoxA(nullptr, e.what(), "Arvo Studio", MB_ICONERROR);
        return 1;
    }
}
#elif defined(__APPLE__)
int RunMacEditor();
int main() { return RunMacEditor(); }
#else
#include <iostream>
int main() { std::cerr << "Use ArvoRender for headless rendering on this platform.\n"; return 1; }
#endif
