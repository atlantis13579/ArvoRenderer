#include "Core/Paths.h"
#include "Editor/ModuleAdapters.h"
#include "Editor/EditorPresenter.h"
#include "Editor/RenderPanel.h"
#include "Editor/Imgui.h"
#include <algorithm>
#include <chrono>
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
    bool DrawingFrame = false;
    bool ClosePending = false;
    bool Closing = false;
    bool ResizePending = false;
};

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

    // Keep the current presentation resources alive while the selected preview initializes.
    if (state.Presenter != nullptr && state.Editor->ViewportBackend() != selectedBackend)
    {
        return true;
    }
    const bool hadPreviousPresenter = state.Presenter != nullptr;
    const Arvo::Editor::EditorPresentationAPI previousAPI = hadPreviousPresenter
        ? state.Presenter->API()
        : Arvo::Editor::EditorPresentationAPI::DX11;
    if (state.Presenter != nullptr)
    {
        // A Win32 window can only own one active flip-model swap chain. Release
        // it before initializing the newly selected native presentation API.
        state.Presenter->Shutdown();
        state.Presenter.reset();
    }

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

bool DrawEditorFrame(WindowState& state)
{
    // Native swapchain operations can synchronously dispatch window messages.
    // Never replace a presenter or begin another frame while one is recording.
    if (state.DrawingFrame) return true;
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
    const auto resetFailedPresenter = [&state](const std::string& message)
    {
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
    if (!state.Presenter->EndFrame())
    {
        return resetFailedPresenter(state.Presenter->LastError());
    }
    return true;
}

void RenderEditorWindow(WindowState& state)
{
    if (state.Closing || IsIconic(static_cast<HWND>(state.NativeWindow))) return;
    (void)DrawEditorFrame(state);
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
    case WM_DPICHANGED:
    {
        const auto* bounds = reinterpret_cast<const RECT*>(lParam);
        SetWindowPos(window, nullptr, bounds->left, bounds->top,
                     bounds->right - bounds->left, bounds->bottom - bounds->top,
                     SWP_NOACTIVATE | SWP_NOZORDER);
        return 0;
    }
    case WM_TIMER:
    {
        const auto now = std::chrono::steady_clock::now();
        const float delta = std::chrono::duration<float>(now - state->LastTick).count();
        state->LastTick = now;
        state->Editor->Tick(std::min(delta, 0.1f));
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
        PostQuitMessage(0);
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

int RunWindow(HINSTANCE instance, int showCommand, Arvo::Editor::DemoId demo)
{
    WindowState state;
    state.Editor = CreateEditor();
    state.Panel.StartLocalWorkers();
    state.Editor->SelectDemo(demo);

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
    ShowWindow(window, showCommand);
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
        // Keep client coordinates in physical pixels; Windows bitmap scaling
        // would otherwise blur even a full-resolution DXR viewport on HiDPI displays.
        if (!SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2))
            throw std::runtime_error("Cannot enable native-pixel editor presentation");
        int count = 0;
        LPWSTR* args = CommandLineToArgvW(GetCommandLineW(), &count);
        auto demo = Arvo::Editor::DemoId::CornellBox;
        for (int i = 1; i < count; ++i)
        {
            if (std::wstring(args[i]) == L"--scene" && i + 1 < count)
            {
                const std::wstring key = args[++i];
                if (!Arvo::Editor::ParseDemo(std::string(key.begin(), key.end()), demo))
                {
                    LocalFree(args);
                    throw std::invalid_argument("Unknown editor scene");
                }
            }
        }
        LocalFree(args);
        return RunWindow(instance, showCommand, demo);
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
int main() { std::cerr << "The interactive editor requires Windows or macOS. Use ArvoWorker for headless rendering.\n"; return 1; }
#endif
