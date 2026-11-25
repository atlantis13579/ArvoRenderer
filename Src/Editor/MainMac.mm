#include "Core/Paths.h"
#include "Editor/Editor.hpp"
#include "Editor/EditorPresenter.h"
#include "Editor/Imgui.h"
#include "Editor/ModuleAdapters.h"
#include "Editor/RenderPanel.h"

#if defined(__APPLE__)

#import <Cocoa/Cocoa.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace
{
using Arvo::Editor::BackendDescriptor;
using Arvo::Editor::DemoDescriptor;
using Arvo::Editor::DemoId;
using Arvo::Editor::EditorCore;
using Arvo::Editor::EditorPresentationAPI;
using Arvo::Editor::FillMode;
using Arvo::Editor::IEditorPresenter;
using Arvo::Editor::Rect;

constexpr int EditorPanelX = 10;
constexpr int EditorPanelWidth = 330;
constexpr int EditorPanelGap = 10;
constexpr int EditorViewportMargin = 10;
constexpr int InitialWidth = 1280;
constexpr int InitialHeight = 720;

std::filesystem::path DefaultContentRoot()
{
    return Arvo::Core::ContentRoot();
}

Rect EditorViewportBounds(int clientWidth, int clientHeight)
{
    const float left = static_cast<float>(EditorPanelX + EditorPanelWidth + EditorPanelGap);
    const float top = static_cast<float>(EditorViewportMargin);
    const float availableWidth = std::max(1.0f, static_cast<float>(clientWidth) - left - EditorViewportMargin);
    const float availableHeight = std::max(1.0f, static_cast<float>(clientHeight) - top - EditorViewportMargin);
    const float width = std::min(availableWidth, availableHeight * 16.0f / 9.0f);
    const float height = width * 9.0f / 16.0f;
    return {left + (availableWidth - width) * 0.5f, top + (availableHeight - height) * 0.5f, width, height};
}

NSString* StringFromUtf8(const std::string& text)
{
    NSString* converted = [NSString stringWithUTF8String:text.c_str()];
    return converted != nil ? converted : @"Unknown Arvo error";
}

// Startup diagnostics go to stderr so a launch that never reaches the window
// server still leaves a trace in the terminal or the system log.
void ArvoLog(const char* format, ...)
{
    std::fputs("[arvo] ", stderr);
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

} // namespace

@interface ArvoEditorView : NSView
{
@private
    std::unique_ptr<EditorCore> Editor_;
    Arvo::Editor::RenderPanel RenderPanel_;
    std::unique_ptr<IEditorPresenter> Presenter_;
    NSTimer* FrameTimer_;
    std::chrono::steady_clock::time_point LastTick_;
    std::string PresentationError_;
    int ClientWidth_;
    int ClientHeight_;
    int MouseX_;
    int MouseY_;
    int MouseWheel_;
    int PanelScroll_;
    double WheelRemainder_;
    unsigned char MouseButtons_;
    BOOL CameraRotateCapture_;
    BOOL Running_;
}

- (BOOL)startEditor;
- (void)stopEditor;
- (NSString*)startupErrorText;
- (void)renderNow;
- (void)resizeEditor;
- (void)drawEditorImgui;
- (void)frameTimer:(NSTimer*)timer;
- (void)updateMouseFromEvent:(NSEvent*)event;

@end

@implementation ArvoEditorView

- (instancetype)initWithFrame:(NSRect)frameRect
{
    self = [super initWithFrame:frameRect];
    if (self != nil)
    {
        FrameTimer_ = nil;
        LastTick_ = std::chrono::steady_clock::now();
        ClientWidth_ = std::max(1, static_cast<int>(std::lround(NSWidth(frameRect))));
        ClientHeight_ = std::max(1, static_cast<int>(std::lround(NSHeight(frameRect))));
        MouseX_ = 0;
        MouseY_ = 0;
        MouseWheel_ = 0;
        PanelScroll_ = 0;
        WheelRemainder_ = 0.0;
        MouseButtons_ = 0;
        CameraRotateCapture_ = NO;
        Running_ = NO;
    }
    return self;
}

- (void)dealloc
{
    [self stopEditor];
    [super dealloc];
}

- (BOOL)acceptsFirstResponder
{
    return YES;
}

- (BOOL)acceptsFirstMouse:(NSEvent*)event
{
    (void)event;
    return YES;
}

- (void)viewDidMoveToWindow
{
    [super viewDidMoveToWindow];
    if ([self window] != nil)
    {
        [[self window] setAcceptsMouseMovedEvents:YES];
    }
    [self resizeEditor];
}

- (void)viewDidChangeBackingProperties
{
    [super viewDidChangeBackingProperties];
    [self resizeEditor];
}

- (void)setFrameSize:(NSSize)newSize
{
    [super setFrameSize:newSize];
    [self resizeEditor];
}

- (BOOL)startEditor
{
    if (Running_)
    {
        return YES;
    }
    if (![NSThread isMainThread])
    {
        PresentationError_ = "The macOS editor must start on the AppKit main thread.";
        return NO;
    }

    [self resizeEditor];
    try
    {
        std::unique_ptr<IEditorPresenter> presenter =
            Arvo::Editor::CreateEditorPresenter(EditorPresentationAPI::Metal);
        if (presenter == nullptr)
        {
            PresentationError_ = "Metal editor presentation is unavailable.";
            ArvoLog("startEditor: CreateEditorPresenter returned null");
            return NO;
        }
        if (!presenter->Initialize(self, ClientWidth_, ClientHeight_))
        {
            PresentationError_ = presenter->LastError();
            ArvoLog("startEditor: presenter Initialize failed: %s", PresentationError_.c_str());
            return NO;
        }

        auto editor = std::make_unique<EditorCore>(DefaultContentRoot());
        Arvo::Editor::RegisterModuleAdapters(*editor);
        editor->SetPickHandler(Arvo::Editor::CreatePickHandler());

        // Metal presents the editor UI on macOS. Scene transport is offline-only;
        // the DX11/DXR realtime preview choices require Windows.
        const Arvo::Editor::Rect viewport = EditorViewportBounds(ClientWidth_, ClientHeight_);
        editor->ResizeViewport(
            static_cast<std::uint32_t>(std::max(viewport.Width, 1.0f)),
            static_cast<std::uint32_t>(std::max(viewport.Height, 1.0f)));

        Presenter_ = std::move(presenter);
        Editor_ = std::move(editor);
        RenderPanel_.StartLocalWorkers();
        PresentationError_.clear();
        LastTick_ = std::chrono::steady_clock::now();
        Running_ = YES;

        // NSTimer has no -initWithTimeInterval:target:selector:userInfo:repeats:
        // instance initializer; that spelling is a class factory. A timer built
        // with the factory is autoreleased, so retain it for the frame loop.
        FrameTimer_ = [[NSTimer timerWithTimeInterval:1.0 / 60.0
            target:self
            selector:@selector(frameTimer:)
            userInfo:nil
            repeats:YES] retain];
        [[NSRunLoop mainRunLoop] addTimer:FrameTimer_ forMode:NSRunLoopCommonModes];
        ArvoLog("editor running at %dx%d", ClientWidth_, ClientHeight_);
        return YES;
    }
    catch (const std::exception& exception)
    {
        PresentationError_ = exception.what();
        ArvoLog("startEditor: exception: %s", PresentationError_.c_str());
    }
    catch (NSException* exception)
    {
        PresentationError_ = "Objective-C exception: " + std::string([[exception name] UTF8String]) +
            ": " + std::string([[exception reason] UTF8String]);
        ArvoLog("startEditor: %s", PresentationError_.c_str());
    }
    catch (...)
    {
        PresentationError_ = "Unknown macOS editor initialization failure.";
        ArvoLog("startEditor: unknown exception");
    }
    // Partial startup may already have taken ownership of the presenter, the
    // worker pool and the editor, so unwind through the normal shutdown path.
    [self stopEditor];
    return NO;
}

- (void)stopEditor
{
    RenderPanel_.StopLocalWorkers();
    Running_ = NO;
    if (Editor_ != nullptr)
    {
        Editor_->ClearCameraInput();
    }
    if (FrameTimer_ != nil)
    {
        [FrameTimer_ invalidate];
        [FrameTimer_ release];
        FrameTimer_ = nil;
    }
    if (Presenter_ != nullptr)
    {
        Presenter_->Shutdown();
        Presenter_.reset();
    }
    Editor_.reset();
}

- (NSString*)startupErrorText
{
    return StringFromUtf8(PresentationError_);
}

- (void)resizeEditor
{
    const NSRect bounds = [self bounds];
    ClientWidth_ = std::max(1, static_cast<int>(std::lround(NSWidth(bounds))));
    ClientHeight_ = std::max(1, static_cast<int>(std::lround(NSHeight(bounds))));
    MouseX_ = std::clamp(MouseX_, 0, ClientWidth_ - 1);
    MouseY_ = std::clamp(MouseY_, 0, ClientHeight_ - 1);
    if (Presenter_ != nullptr && Presenter_->IsInitialized() &&
        !Presenter_->Resize(ClientWidth_, ClientHeight_))
    {
        PresentationError_ = Presenter_->LastError();
    }
    if (Editor_ != nullptr)
    {
        const Arvo::Editor::Rect viewport = EditorViewportBounds(ClientWidth_, ClientHeight_);
        Editor_->ResizeViewport(
            static_cast<std::uint32_t>(std::max(viewport.Width, 1.0f)),
            static_cast<std::uint32_t>(std::max(viewport.Height, 1.0f)));
    }
}

- (void)drawEditorImgui
{
    if (Editor_ != nullptr)
    {
        RenderPanel_.Draw(*Editor_, ClientHeight_);
        if (imguiTextFocused()) Editor_->ClearCameraInput();
    }
}

- (void)renderNow
{
    if (!Running_ || Editor_ == nullptr || Presenter_ == nullptr ||
        !Presenter_->IsInitialized())
    {
        return;
    }
    NSWindow* window = [self window];
    if (window == nil || ![window isVisible] || [window isMiniaturized] ||
        ([window occlusionState] & NSWindowOcclusionStateVisible) == 0)
    {
        return;
    }

    const Arvo::Editor::Rect viewport = EditorViewportBounds(ClientWidth_, ClientHeight_);
    if (!Presenter_->BeginFrame(
            Editor_->ViewportImage(),
            Editor_->ViewportRevision(),
            static_cast<int>(viewport.X),
            static_cast<int>(viewport.Y),
            static_cast<int>(viewport.Width),
            static_cast<int>(viewport.Height)))
    {
        PresentationError_ = Presenter_->LastError();
        return;
    }

    // AppKit events use a bottom-left origin. MouseY_ is kept in the
    // top-left viewport convention and converted back for Riemann ImGui.
    imguiBeginFrame(
        MouseX_,
        ClientHeight_ - 1 - MouseY_,
        MouseButtons_,
        MouseWheel_);
    MouseWheel_ = 0;
    [self drawEditorImgui];
    imguiEndFrame();
    Presenter_->DrawImgui();
    if (!Presenter_->EndFrame())
    {
        PresentationError_ = Presenter_->LastError();
        ArvoLog("renderNow: EndFrame failed: %s", PresentationError_.c_str());
        return;
    }
    PresentationError_.clear();
}

- (void)frameTimer:(NSTimer*)timer
{
    (void)timer;
    if (!Running_ || Editor_ == nullptr)
    {
        return;
    }

    NSAutoreleasePool* framePool = [[NSAutoreleasePool alloc] init];
    const auto now = std::chrono::steady_clock::now();
    const float deltaSeconds = std::clamp(
        std::chrono::duration<float>(now - LastTick_).count(),
        0.0f,
        0.1f);
    LastTick_ = now;

    // Tick only polls futures and advances realtime scenes; it never waits for
    // an offline renderer. Completed images are uploaded during presentation.
    Editor_->Tick(deltaSeconds);
    [self renderNow];
    [framePool drain];
}

- (void)updateMouseFromEvent:(NSEvent*)event
{
    const NSPoint point = [self convertPoint:[event locationInWindow] fromView:nil];
    MouseX_ = std::clamp(
        static_cast<int>(std::lround(point.x)),
        0,
        std::max(ClientWidth_ - 1, 0));
    MouseY_ = std::clamp(
        static_cast<int>(std::lround(ClientHeight_ - 1.0 - point.y)),
        0,
        std::max(ClientHeight_ - 1, 0));
}

- (void)mouseMoved:(NSEvent*)event
{
    [self updateMouseFromEvent:event];
}

- (void)mouseDragged:(NSEvent*)event
{
    [self updateMouseFromEvent:event];
}

- (void)rightMouseDragged:(NSEvent*)event
{
    [self updateMouseFromEvent:event];
    if (Editor_ != nullptr)
    {
        Editor_->HandleCameraPointer(
            static_cast<float>(MouseX_),
            static_cast<float>(MouseY_),
            CameraRotateCapture_ == YES);
    }
}

- (void)mouseDown:(NSEvent*)event
{
    [self updateMouseFromEvent:event];
    MouseButtons_ |= IMGUI_MBUT_LEFT;
    if (Editor_ != nullptr)
    {
        const Arvo::Editor::Rect viewport = EditorViewportBounds(ClientWidth_, ClientHeight_);
        Editor_->HandleViewportPointer(
            static_cast<float>(MouseX_),
            static_cast<float>(MouseY_),
            viewport);
    }
    [self renderNow];
}

- (void)mouseUp:(NSEvent*)event
{
    [self updateMouseFromEvent:event];
    MouseButtons_ &= static_cast<unsigned char>(~IMGUI_MBUT_LEFT);
    [self renderNow];
}

- (void)rightMouseDown:(NSEvent*)event
{
    [self updateMouseFromEvent:event];
    MouseButtons_ |= IMGUI_MBUT_RIGHT;
    const Arvo::Editor::Rect viewport = EditorViewportBounds(ClientWidth_, ClientHeight_);
    CameraRotateCapture_ = viewport.Contains(
        static_cast<float>(MouseX_),
        static_cast<float>(MouseY_));
    if (Editor_ != nullptr)
    {
        Editor_->HandleCameraPointer(
            static_cast<float>(MouseX_),
            static_cast<float>(MouseY_),
            CameraRotateCapture_ == YES);
    }
    [[self window] makeFirstResponder:self];
    [self renderNow];
}

- (void)rightMouseUp:(NSEvent*)event
{
    [self updateMouseFromEvent:event];
    MouseButtons_ &= static_cast<unsigned char>(~IMGUI_MBUT_RIGHT);
    CameraRotateCapture_ = NO;
    if (Editor_ != nullptr)
    {
        Editor_->HandleCameraPointer(
            static_cast<float>(MouseX_),
            static_cast<float>(MouseY_),
            false);
    }
    [self renderNow];
}

- (void)keyDown:(NSEvent*)event
{
    if (imguiTextFocused())
    {
        NSString* text = [event characters];
        for (NSUInteger i = 0; i < [text length]; ++i) imguiTextCharacter([text characterAtIndex:i]);
        return;
    }
    if (Editor_ == nullptr)
    {
        [super keyDown:event];
        return;
    }
    NSString* characters = [event charactersIgnoringModifiers];
    if ([characters length] != 1)
    {
        [super keyDown:event];
        return;
    }
    const unichar key = [characters characterAtIndex:0];
    if (key == 'w' || key == 'W' || key == 'a' || key == 'A' ||
        key == 's' || key == 'S' || key == 'd' || key == 'D')
    {
        Editor_->SetCameraMovementKey(static_cast<char>(key), true);
        return;
    }
    [super keyDown:event];
}

- (void)keyUp:(NSEvent*)event
{
    if (Editor_ == nullptr)
    {
        [super keyUp:event];
        return;
    }
    NSString* characters = [event charactersIgnoringModifiers];
    if ([characters length] != 1)
    {
        [super keyUp:event];
        return;
    }
    const unichar key = [characters characterAtIndex:0];
    if (key == 'w' || key == 'W' || key == 'a' || key == 'A' ||
        key == 's' || key == 'S' || key == 'd' || key == 'D')
    {
        Editor_->SetCameraMovementKey(static_cast<char>(key), false);
        return;
    }
    [super keyUp:event];
}

- (BOOL)resignFirstResponder
{
    CameraRotateCapture_ = NO;
    MouseButtons_ &= static_cast<unsigned char>(~IMGUI_MBUT_RIGHT);
    if (Editor_ != nullptr)
    {
        Editor_->ClearCameraInput();
    }
    return [super resignFirstResponder];
}

- (void)scrollWheel:(NSEvent*)event
{
    [self updateMouseFromEvent:event];
    WheelRemainder_ += [event scrollingDeltaY];
    const int wheelSteps = static_cast<int>(WheelRemainder_);
    if (wheelSteps != 0)
    {
        MouseWheel_ = std::clamp(MouseWheel_ + wheelSteps, -64, 64);
        WheelRemainder_ -= wheelSteps;
        [self renderNow];
    }
}

@end

@interface ArvoApplicationDelegate : NSObject <NSApplicationDelegate>
{
@private
    NSWindow* Window_;
    ArvoEditorView* EditorView_;
    int ExitCode_;
}

- (int)exitCode;

@end


@implementation ArvoApplicationDelegate

- (instancetype)init
{
    self = [super init];
    if (self != nil)
    {
        Window_ = nil;
        EditorView_ = nil;
        ExitCode_ = 0;
    }
    return self;
}

- (void)dealloc
{
    if (EditorView_ != nil)
    {
        [EditorView_ stopEditor];
    }
    [EditorView_ release];
    [Window_ release];
    [super dealloc];
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification
{
    (void)notification;
    ArvoLog("applicationDidFinishLaunching");
    const NSWindowStyleMask style =
        NSWindowStyleMaskTitled |
        NSWindowStyleMaskClosable |
        NSWindowStyleMaskMiniaturizable |
        NSWindowStyleMaskResizable;
    Window_ = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(0.0, 0.0, InitialWidth, InitialHeight)
        styleMask:style
        backing:NSBackingStoreBuffered
        defer:NO];
    [Window_ setTitle:@"Arvo Editor"];
    [Window_ setContentMinSize:NSMakeSize(720.0, 480.0)];
    [Window_ setReleasedWhenClosed:NO];

    EditorView_ = [[ArvoEditorView alloc]
        initWithFrame:[[Window_ contentView] bounds]];
    [EditorView_ setAutoresizingMask:NSViewWidthSizable | NSViewHeightSizable];
    [Window_ setContentView:EditorView_];
    [Window_ makeFirstResponder:EditorView_];

    if (![EditorView_ startEditor])
    {
        ExitCode_ = 2;
        ArvoLog("startup failed: %s", [[EditorView_ startupErrorText] UTF8String]);
        NSAlert* alert = [[NSAlert alloc] init];
        [alert setMessageText:@"Arvo Editor could not start"];
        [alert setInformativeText:[EditorView_ startupErrorText]];
        [alert addButtonWithTitle:@"Quit"];
        [alert runModal];
        [alert release];
        [NSApp terminate:nil];
        return;
    }

    [Window_ center];
    [Window_ makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
    [EditorView_ renderNow];
}

- (void)applicationWillTerminate:(NSNotification*)notification
{
    (void)notification;
    [EditorView_ stopEditor];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender
{
    (void)sender;
    return YES;
}

- (BOOL)applicationShouldHandleReopen:(NSApplication*)sender
                    hasVisibleWindows:(BOOL)hasVisibleWindows
{
    (void)sender;
    if (!hasVisibleWindows && Window_ != nil)
    {
        [Window_ makeKeyAndOrderFront:nil];
    }
    return YES;
}

- (int)exitCode
{
    return ExitCode_;
}

@end

namespace
{

void InstallApplicationMenu(NSApplication* application)
{
    NSMenu* menuBar = [[NSMenu alloc] initWithTitle:@""];
    NSMenuItem* applicationItem = [[NSMenuItem alloc]
        initWithTitle:@""
        action:nil
        keyEquivalent:@""];
    [menuBar addItem:applicationItem];

    NSMenu* applicationMenu = [[NSMenu alloc] initWithTitle:@"Arvo"];
    NSMenuItem* quitItem = [[NSMenuItem alloc]
        initWithTitle:@"Quit Arvo"
        action:@selector(terminate:)
        keyEquivalent:@"q"];
    [applicationMenu addItem:quitItem];
    [applicationItem setSubmenu:applicationMenu];
    [application setMainMenu:menuBar];

    [quitItem release];
    [applicationMenu release];
    [applicationItem release];
    [menuBar release];
}

} // namespace

int RunMacEditor()
{
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    NSApplication* application = [NSApplication sharedApplication];
    [application setActivationPolicy:NSApplicationActivationPolicyRegular];
    InstallApplicationMenu(application);

    ArvoApplicationDelegate* delegate =
        [[ArvoApplicationDelegate alloc] init];
    [application setDelegate:delegate];
    [application run];
    const int exitCode = [delegate exitCode];
    [application setDelegate:nil];
    [delegate release];
    [pool drain];
    return exitCode;
}

#endif
