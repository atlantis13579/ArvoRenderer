#pragma once
#include "Renderer/Reconstruction.h"
namespace Arvo::RayTracing::Cpu
{
bool SpatialDenoise(const LinearFilm &, const ReconstructionOptions &, bool useMoments, bool gammaCorrection,
                    const Detail::CancellationProbe &, ReconstructedImage &, std::string &error,
                    const GeometryGuideImage *guides = nullptr);
}
