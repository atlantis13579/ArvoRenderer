#pragma once
#include <cstdint>

namespace Arvo::RayTracing::CudaPhoton
{
// Upload/storage ABI only. The independently authored .cu owns all transport
// and hash-grid operations; the driver never builds or queries a CPU photon map.
inline constexpr unsigned MaximumReflections = 32, ThreadsPerBlock = 128;
inline constexpr unsigned EmissionBatch = 1024, CameraBatch = 1024, IndexBatch = 65536;
inline constexpr unsigned Invalid = 0xffffffffu, NumericalFailure = 1, InvalidConfiguration = 2;
inline constexpr unsigned Direct = 1, Caustic = 2, Indirect = 3;
// Volume vertices keep their material identity in the existing 64-byte slot.
inline constexpr unsigned VolumeFlag = 0x80000000u, ClassMask = ~VolumeFlag;
inline constexpr unsigned CausticEnabled = 1, IndirectEnabled = 2;
struct Vector
{
    float X, Y, Z;
};
struct alignas(16) Photon
{
    Vector Position;
    unsigned Next;
    Vector Incoming;
    unsigned LightDepth;
    Vector Flux;
    unsigned Kind;
    Vector Geometry;
    unsigned MaterialIndex;
};
struct alignas(16) Controls
{
    Vector WorldCenter;
    float WorldRadius;
    double Radius;
    unsigned EmitterCount, PhotonPaths;
    unsigned SlotsPerPath, BucketCount, Seed, Classes;
    unsigned Begin, Count, Sample, Reserved;
};
static_assert(sizeof(Photon) == 64 && sizeof(Controls) == 64);
} // namespace Arvo::RayTracing::CudaPhoton
