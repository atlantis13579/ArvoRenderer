#pragma once
#include <cstdint>
namespace Arvo::RayTracing::CudaPpm
{
// Data contracts only; the independently handwritten .cu owns all transport.
constexpr unsigned MaximumReflections = 32, ThreadsPerBlock = 128;
constexpr unsigned TransportBatch = 1024, LinearBatch = 65536;
constexpr unsigned Invalid = 0xffffffffu, NumericalError = 1, ConfigurationError = 2, IndexError = 4;
constexpr unsigned Caustics = 1, Indirect = 2;
enum VisibleKind : unsigned { VisibleNone, VisibleSurface, VisibleVolume };
struct Vector
{
    float X, Y, Z;
};
struct alignas(16) VisiblePoint
{
    Vector Position;
    unsigned Active;
    Vector Geometry;
    unsigned Depth;
    Vector Normal;
    unsigned FrontFace;
    Vector View;
    unsigned Material;
    Vector Color;
    unsigned Next;
    Vector Throughput;
    unsigned Reserved;
};
struct alignas(16) Pixel
{
    double RadiusSquared, EffectiveCount;
    double Flux[3], Direct[3];
    double Incoming[3];
    unsigned Arrivals, Reserved;
    double VolumeRadiusSquared, VolumeEffectiveCount;
    double VolumeFlux[3], VolumeIncoming[3];
    unsigned VolumeArrivals, VolumeReserved[3];
};
struct alignas(16) Controls
{
    Vector Center;
    float WorldRadius;
    double InitialRadius, Alpha;
    unsigned Sources, PathsPerPass, Buckets, Seed;
    unsigned Classes, Begin, Count, Pass;
};
static_assert(sizeof(VisiblePoint) == 96 && sizeof(Pixel) == 176 && sizeof(Controls) == 64);
} // namespace Arvo::RayTracing::CudaPpm
