#pragma once
#include "Renderer/RayTracing.h"
#include <functional>
#include <memory>
#include <vector>

namespace Arvo::RayTracing::Cpu
{
// CPU spatial lookup only. Integrators own photon/visible-point payloads and
// kernel estimators; this index never evaluates radiance or gathers a fixed k.
class PointKdTree
{
  public:
    using Visitor = std::function<void(uint32_t index, double squaredDistance)>;
    using Cancellation = std::function<bool()>;
    explicit PointKdTree(std::vector<Float3> positions, Cancellation cancellation = {});
    ~PointKdTree();
    PointKdTree(PointKdTree&&) noexcept;
    PointKdTree& operator=(PointKdTree&&) noexcept;
    size_t Size() const;
    bool WasCancelled() const;
    void Search(Float3 center, double squaredRadius, const Visitor&) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> Data;
};
} // namespace Arvo::RayTracing::Cpu
