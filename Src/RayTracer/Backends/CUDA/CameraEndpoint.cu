#include <cuda_runtime.h>
namespace ArvoCudaCamera
{
// Independent CUDA camera endpoint arithmetic. PBRT 3e, "Sampling Cameras"
// (section 16.1.1). A full-film area sample yields p(w)=1/(A*cos^3), where
// A=4*HalfWidth*HalfHeight. We=p(lens)*p(w)/cos preserves aperture averaging.
// Raster spans the FULL image, not a tile. Pixel/splat normalization is external.
constexpr float Pi = 3.14159265358979323846f;
constexpr unsigned Valid = 1, DeltaPosition = 2, DeltaDirection = 4;
struct Frame
{
    float3 Position, Forward, Right, Up;
    float HalfWidth, HalfHeight, LensRadius, FocusDistance;
};
struct Evaluation
{
    float2 Raster;
    float Importance, PositionPdf, DirectionPdf;
    unsigned Flags;
};
struct Sample
{
    float3 Origin, Direction;
    float2 LensPoint, Raster;
    float Importance, PositionPdf, DirectionPdf, Weight;
    unsigned Flags;
};
struct Connection
{
    float3 Position, Direction;
    float2 LensPoint, Raster;
    float Distance, Weight, PositionPdf, DirectionPdf;
    unsigned Flags;
};
__device__ float3 Plus(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 Minus(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 Scale(float3 v, float s)
{
    return make_float3(v.x * s, v.y * s, v.z * s);
}
__device__ float3 Divide(float3 v, float s)
{
    return make_float3(v.x / s, v.y / s, v.z / s);
}
__device__ float Dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float Length(float3 v)
{
    float peak = fmaxf(fabsf(v.x), fmaxf(fabsf(v.y), fabsf(v.z)));
    if (!(peak > 0) || !isfinite(peak))
        return 0;
    float3 scaled = Divide(v, peak);
    return peak * sqrtf(Dot(scaled, scaled));
}
__device__ bool InSquare(float2 v)
{
    return v.x >= 0 && v.x < 1 && v.y >= 0 && v.y < 1;
}
__device__ float2 PixelRaster(unsigned x, unsigned y, unsigned width, unsigned height, float2 jitter)
{
    if (x >= width || y >= height || !InSquare(jitter))
        return make_float2(-1, -1);
    // A valid last-pixel sample can round to 1. Preserve the camera's half-open
    // support after conversion without relaxing validation of external rasters.
    const float upper = nextafterf(1.0f, 0.0f);
    return make_float2(fminf(float((double(x) + jitter.x) / width), upper),
                       fminf(float((double(y) + jitter.y) / height), upper));
}
__device__ float LensPdf(const Frame& frame)
{
    if (!(frame.HalfWidth > 0 && frame.HalfHeight > 0 && frame.LensRadius >= 0))
        return 0;
    if (frame.LensRadius == 0)
        return 1;
    if (!(frame.FocusDistance > 0))
        return 0;
    float value = 1 / (Pi * frame.LensRadius * frame.LensRadius);
    return value > 0 && isfinite(value) ? value : 0;
}
__device__ float2 Aperture(const Frame& frame, float2 u)
{
    float horizontal = 2 * u.x - 1, vertical = 2 * u.y - 1;
    if (frame.LensRadius == 0 || (horizontal == 0 && vertical == 0))
        return make_float2(0, 0);
    float radius, angle;
    if (fabsf(horizontal) > fabsf(vertical))
    {
        radius = horizontal;
        angle = (Pi / 4) * vertical / horizontal;
    }
    else
    {
        radius = vertical;
        angle = Pi / 2 - (Pi / 4) * horizontal / vertical;
    }
    return make_float2(frame.LensRadius * radius * cosf(angle), frame.LensRadius * radius * sinf(angle));
}
__device__ Evaluation Response(const Frame& frame, float2 raster, float cosine)
{
    Evaluation response{};
    float pLens = LensPdf(frame);
    if (!(pLens > 0 && cosine > 0))
        return response;
    float area = 4 * frame.HalfWidth * frame.HalfHeight;
    float pDirection = 1 / (area * cosine * cosine * cosine), importance = pLens * pDirection / cosine;
    if (!(pDirection > 0 && importance > 0) || !isfinite(pDirection) || !isfinite(importance))
        return response;
    response.Raster = raster;
    response.Importance = importance;
    response.DirectionPdf = pDirection;
    response.PositionPdf = frame.LensRadius == 0 ? 0 : pLens;
    response.Flags = Valid | (frame.LensRadius == 0 ? DeltaPosition : 0);
    return response;
}
// Frame is orthonormal, direction unit and outgoing. Lens coordinates already
// lie on aperture support, in world units along Right/Up (zero for a pinhole).
__device__ Evaluation Evaluate(const Frame& frame, float2 lens, float3 direction)
{
    float cosine = Dot(frame.Forward, direction);
    if (!(cosine > 0) || !(LensPdf(frame) > 0))
        return {};
    float focus = frame.LensRadius == 0 ? 1 : frame.FocusDistance;
    float x = lens.x / focus + Dot(direction, frame.Right) / cosine;
    float y = lens.y / focus + Dot(direction, frame.Up) / cosine;
    float2 raster = make_float2(0.5f + x / (2 * frame.HalfWidth), 0.5f - y / (2 * frame.HalfHeight));
    if (!InSquare(raster))
        return {};
    return Response(frame, raster, cosine);
}
__device__ Sample Emit(const Frame& frame, float2 raster, float2 lensU)
{
    if (!InSquare(raster) || !InSquare(lensU) || !(LensPdf(frame) > 0))
        return {};
    float2 lens = Aperture(frame, lensU);
    float3 offset = Plus(Scale(frame.Right, lens.x), Scale(frame.Up, lens.y));
    float3 axis = Plus(Plus(frame.Forward, Scale(frame.Right, (2 * raster.x - 1) * frame.HalfWidth)),
                       Scale(frame.Up, (1 - 2 * raster.y) * frame.HalfHeight));
    float3 toward = frame.LensRadius == 0 ? axis : Minus(Scale(axis, frame.FocusDistance), offset);
    float length = Length(toward);
    if (!(length > 0) || !isfinite(length))
        return {};
    float3 direction = Divide(toward, length);
    Evaluation response = Response(frame, raster, Dot(direction, frame.Forward));
    if (!response.Flags)
        return {};
    return {Plus(frame.Position, offset),
            direction,
            lens,
            raster,
            response.Importance,
            frame.LensRadius == 0 ? 1 : response.PositionPdf,
            response.DirectionPdf,
            1,
            response.Flags};
}
// Return a scene-to-lens direction. No visibility or scene BSDF is evaluated.
// Finite apertures return an angular density. Pinhole connections retain unit
// delta mass and put the lens geometry factor in Weight explicitly.
__device__ Connection Connect(const Frame& frame, float3 reference, float2 variates)
{
    if (!InSquare(variates) || !(LensPdf(frame) > 0))
        return {};
    float2 lens = Aperture(frame, variates);
    float3 offset = Plus(Scale(frame.Right, lens.x), Scale(frame.Up, lens.y));
    float3 segment = Minus(Minus(reference, frame.Position), offset);
    float distance = Length(segment);
    if (!(distance > 0) || !isfinite(distance))
        return {};
    float3 outward = Divide(segment, distance);
    Evaluation response = Evaluate(frame, lens, outward);
    if (!response.Flags)
        return {};
    bool pinhole = frame.LensRadius == 0;
    float cosine = Dot(frame.Forward, outward);
    float pdf = pinhole ? 1 : (response.PositionPdf * distance) * (distance / cosine);
    float weight = response.DirectionPdf / distance / distance;
    if (!(pdf > 0 && weight > 0) || !isfinite(pdf) || !isfinite(weight))
        return {};
    return {Plus(frame.Position, offset),
            Scale(outward, -1),
            lens,
            response.Raster,
            distance,
            weight,
            pinhole ? 1 : response.PositionPdf,
            pdf,
            response.Flags | (pinhole ? DeltaDirection : 0)};
}
} // namespace ArvoCudaCamera
