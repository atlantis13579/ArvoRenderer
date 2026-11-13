#pragma once
#include "Renderer/RayTracing.h"

namespace Arvo::RayTracing::Cpu
{
enum class EmissionKind : uint32_t
{
    Directional,
    Point,
    Rectangle,
    Environment
};
enum EmissionFlag : uint32_t
{
    EmissionValid = 1,
    EmissionDeltaPosition = 2,
    EmissionDeltaDirection = 4,
    EmissionInfinite = 8
};
struct EmissionDomain
{
    Float3 Center;
    float Radius = 0;
};
struct EmissionSource
{
    EmissionKind Kind = EmissionKind::Point;
    Float3 Position, Direction, Strength;
    Float3 BackgroundTop, BackgroundBottom;
    Float2 Size;
};
struct EmissionSample
{
    Float3 Origin, Direction, Normal, Value, Weight;
    float PositionPdf = 0, DirectionPdf = 0;
    uint32_t Flags = 0;
};
struct EmissionDensity
{
    float Position = 0, Direction = 0;
};

// Conditional on source selection; the caller divides Weight by the source PMF.
// Source fields are finite/nonnegative, axes nonzero, variates in [0,1).
// Direction means direction TO a distant light, or OUT of a rectangular light.
// Domain is a strictly enclosing world-space scene sphere for infinite sources.
EmissionSample SampleLightEmission(const EmissionSource& source, const EmissionDomain& domain,
                                   Float2 positionVariates, Float2 directionVariates);
// Continuous densities only: delta dimensions return zero, while Sample uses
// unit probability masses there. The position is assumed on the source support
// (the direction-dependent disk for infinite lights); direction is unit length.
EmissionDensity LightEmissionPdf(const EmissionSource& source, const EmissionDomain& domain,
                                 Float3 direction);
} // namespace Arvo::RayTracing::Cpu
