#pragma once
#include "Renderer/TemporalReconstruction.h"

namespace Arvo::RayTracing
{
// A sequence owner, not a renderer backend. Film generation, geometry queries and
// temporal arithmetic still execute in the selected native RHI.
class ARVO_RENDERER_API TemporalSession
{
    struct State;
    struct Prepared;

  public:
    class Frame
    {
      public:
        Frame() = default;
        const RenderSettings& Settings() const;
        const TemporalIdentity& Identity() const;

      private:
        std::shared_ptr<const Prepared> Data_;
        friend class TemporalSession;
    };
    struct Result
    {
        RenderStatus Status = RenderStatus::Failed;
        Image Display;
        LinearFilm Film;
        RenderStatistics Statistics;
        TemporalIdentity Identity;
        TemporalReset Reset = TemporalReset::NoHistory;
        uint64_t AcceptedHistoryPixels = 0;
        std::string Error;
        // Invoke only when the complete output is accepted. Dropping a result
        // discards its candidate history. False means superseded or cancelled.
        std::function<bool()> Accept;
    };

    TemporalSession();
    TemporalSession(const TemporalSession&) = delete;
    TemporalSession& operator=(const TemporalSession&) = delete;

    // ContentRevision must change with geometry, materials or lighting. Camera
    // pose may change without resetting correspondence. Transport changes are
    // detected from explicit settings; thread count is only a scheduling policy.
    Frame Begin(const RenderSettings&, uint64_t contentRevision, const TemporalOptions& = {});
    using Reconstruction = std::function<TemporalResult(const TemporalFrame&, const TemporalHistory*,
                                                       const TemporalOptions&, const CancellationToken&)>;
    // A remote transport can call the selected native worker without pretending
    // to implement the rest of IRayTracingRHI or publishing hidden worker state.
    Result Reconstruct(const Reconstruction&, const Frame&, const LinearFilm&, const GeometryGuideImage&,
                       const Camera&, const CancellationToken& = {});
    Result Reconstruct(IRayTracingRHI&, const Frame&, const LinearFilm&, const GeometryGuideImage&,
                       const Camera&, const CancellationToken& = {});
    Result Render(IRayTracingRHI&, std::shared_ptr<const Scene>, const Camera&, const RenderSettings&,
                  uint64_t contentRevision, const TemporalOptions& = {}, const CancellationToken& = {});
    void Reset();
    std::shared_ptr<const TemporalHistory> History() const;
    // Independent transaction branch; immutable accepted images are shared.
    // Publishing or resetting the branch never changes its parent session.
    std::unique_ptr<TemporalSession> Fork() const;

  private:
    std::shared_ptr<State> State_;
};
} // namespace Arvo::RayTracing
