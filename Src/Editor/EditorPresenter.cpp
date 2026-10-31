#include "Editor/EditorPresenter.h"
#if defined(_WIN32)
#include "Editor/EditorPresenterDX11.h"
#elif defined(__APPLE__)
#include "Editor/EditorPresenterMetal.h"
#endif
namespace Arvo::Editor
{
std::unique_ptr<IEditorPresenter> CreateEditorPresenter(EditorPresentationAPI api)
{
#if defined(_WIN32)
    if (api == EditorPresentationAPI::DX11) return std::make_unique<EditorPresenterDX11>();
#elif defined(__APPLE__)
    if (api == EditorPresentationAPI::Metal) return std::make_unique<EditorPresenterMetal>();
#else
    (void)api;
#endif
    return nullptr;
}
const char* ToString(EditorPresentationAPI api) noexcept
{
    return api == EditorPresentationAPI::Metal ? "Metal" : "DirectX 11";
}
EditorPresentationAPI ResolvePresentationAPI(std::string_view, EditorPresentationAPI previous) noexcept
{
    return previous;
}
}
