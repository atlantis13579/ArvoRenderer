#pragma once

#include "Distributed/OfflineRenderer.h"
#include "Distributed/LocalWorkers.h"
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
    void StartLocalWorkers();
    void StopLocalWorkers();
    void SetAdaptiveSampling(RayTracing::AdaptiveOptions options, std::uint32_t previewLimit);
    void SetAcceleration(RayTracing::AccelerationType type);
    void SetBvhPolicy(RayTracing::BvhBuildPolicy policy);
    void SetGeometryMode(RayTracing::GeometryMode mode);
    void SetTemporalRendering(Distributed::TemporalRenderOptions options);
    const Distributed::TemporalRenderOptions& TemporalRendering() const { return Temporal_; }
    [[nodiscard]] Distributed::Progress Progress() const;
private:
    void PollPreparation();
    void DrawMaterials(EditorCore& editor);
    void DrawCamera(EditorCore& editor);
    void DrawObjectMotion(EditorCore& editor);
    struct PreparedRender
    {
        Distributed::RenderConfig Config;
        Distributed::TemporalRenderOptions Temporal;
    };
    std::shared_ptr<Distributed::LocalWorkers> LocalWorkers_;
    Distributed::OfflineRenderClient Client_;
    std::future<PreparedRender> Preparation_;
    std::shared_ptr<std::atomic<bool>> PreparationCancelled_;
    bool PreparationWasCancelled_ = false;
    std::vector<std::string> Servers_{"http://127.0.0.1:7831", "http://127.0.0.1:7832"};
    std::string Token_;
    std::string OutputDirectory_;
    std::string Error_;
    int Scroll_ = 0;
    int Backend_ = 0;
    int Algorithm_ = 1;
    int Radiometry_ = 0;
    RayTracing::TransportOptions Transport_;
    int Sampling_ = 0;
    uint32_t SamplingSeed_ = 0;
    int Acceleration_ = 0;
    int BvhPolicy_ = 0;
    int Geometry_ = 0;
    bool MaterialsOpen_ = false;
    bool CameraOpen_ = false;
    bool ObjectMotionOpen_ = false;
    int MotionNode_ = 0;
    int Material_ = 0;
    RayTracing::AdaptiveOptions Adaptive_;
    RayTracing::MediumOptions Medium_;
    RayTracing::MicropolygonOptions Micropolygons_;
    bool MediumBoundsOpen_ = false;
    RayTracing::ReconstructionOptions Reconstruction_;
    Distributed::TemporalRenderOptions Temporal_;
    RayTracing::MetropolisOptions Metropolis_;
    RayTracing::PhotonMappingOptions Photons_;
    RayTracing::ProgressivePhotonOptions ProgressivePhotons_;
    RayTracing::VertexConnectionMergingOptions Vcm_;
    RayTracing::PointGiOptions PointGi_;
    int Resolution_ = 2;
    float Samples_ = 16;
    float Bounces_ = 4;
    float TileSize_ = 256;
    float PreviewSamples_ = 1;
};
}
