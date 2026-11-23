#pragma once
#include "Lightmap/Exchange.h"
#include <functional>
namespace Arvo::Lightmap
{
enum class Backend { CUDA, CPU };
// Estimates E/pi with cosine-weighted paths and a Lambertian transport model.
// IndirectOnly excludes analytic light and environment connections at the receiver.
// Emissive geometry is integrated by path hits; no duplicate emission NEE.
ArvoLightmap::Result Bake(const ArvoLightmap::Job&,const std::function<void(uint32_t,uint32_t)>& progress={},Backend backend=Backend::CUDA);
}
