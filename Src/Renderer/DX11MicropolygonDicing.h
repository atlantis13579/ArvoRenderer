#pragma once
#include "Renderer/MicropolygonPreparation.h"
#if defined(_WIN32)
struct ID3D11Device;
struct ID3D11DeviceContext;
namespace Arvo::Renderer
{
// Geometry preparation for raster preview; never registered as an offline RHI.
class DX11MicropolygonDicer
{
  public:
    DX11MicropolygonDicer(ID3D11Device*, ID3D11DeviceContext*);
    ~DX11MicropolygonDicer();
    RayTracing::MicropolygonResult Prepare(const RayTracing::Scene&, const RayTracing::Camera&,
                                           const RayTracing::RenderSettings&,
                                           const RayTracing::CancellationToken&);

  private:
    struct Impl;
    std::unique_ptr<Impl> Impl_;
};
} // namespace Arvo::Renderer
#endif
