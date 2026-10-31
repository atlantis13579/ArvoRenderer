#pragma once

#include "Distributed/Distributed.h"
#include "Editor/Editor.hpp"
#include <atomic>
#include <future>

namespace Arvo::Editor
{
class RenderPanel
{
public:
    RenderPanel();
    ~RenderPanel();
    void Draw(EditorCore& editor, int height);
    void Cancel();
    void Start(EditorCore& editor);
    [[nodiscard]] Distributed::Progress Progress() const;
private:
    void PollPreparation();
    Distributed::RenderClient Client_;
    std::future<Distributed::RenderConfig> Preparation_;
    std::shared_ptr<std::atomic<bool>> PreparationCancelled_;
    bool PreparationWasCancelled_ = false;
    std::vector<std::string> Servers_{"http://127.0.0.1:7831", "http://127.0.0.1:7832"};
    std::string Token_;
    std::string OutputDirectory_;
    std::string Error_;
    int Scroll_ = 0;
    int Backend_ = 2;
    int Algorithm_ = 1;
    int Resolution_ = 2;
    float Samples_ = 16;
    float Bounces_ = 4;
    float TileSize_ = 256;
    float PreviewSamples_ = 1;
};
}
