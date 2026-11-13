// CUDA owns both camera-density planning and displaced triangle generation.
// Components share integer grid levels, and edge interpolation follows stable
// corner IDs across UV/normal seams. See Cook et al., Reyes (SIGGRAPH 1987).
#include <cuda_runtime.h>
#include <cmath>
struct DicingVertex
{
    float4 Position, Normal, Color, TexCoord;
};
struct DicingPatch
{
    DicingVertex Corners[3];
    uint4 Metadata, CornerIds;
};
struct DicingTriangle
{
    DicingVertex Corners[3];
};
struct DicingControls
{
    float4 CameraPositionTarget, CameraForwardHalfHeight, CameraRightHalfWidth, CameraUpAmplitude, Wave;
    uint4 Populations, Extent, Dispatch;
};
constexpr unsigned DicingSmooth = 1, DicingDisplace = 2, DicingClamped = 1u << 8;
constexpr float DicingNearPlane = 1.e-4f, DicingWaveSegments = 8;
constexpr float DicingTwoPi = 2 * 3.14159265358979323846f;
__device__ float3 DicingVector(float4 value)
{
    return make_float3(value.x, value.y, value.z);
}
__device__ float3 DicingAdd(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 DicingSubtract(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 DicingScale(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ float DicingDot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 DicingCross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ unsigned DicingPlan(DicingPatch patch, DicingControls controls)
{
    float3 low = DicingVector(patch.Corners[0].Position), high = low;
    for (unsigned vertex = 1; vertex < 3; ++vertex)
    {
        const float4 p = patch.Corners[vertex].Position;
        low = make_float3(fminf(low.x, p.x), fminf(low.y, p.y), fminf(low.z, p.z));
        high = make_float3(fmaxf(high.x, p.x), fmaxf(high.y, p.y), fmaxf(high.z, p.z));
    }
    const bool displaced = (patch.Metadata.y & DicingDisplace) != 0;
    const float margin = displaced ? fabsf(controls.CameraUpAmplitude.w) : 0;
    low.y -= margin;
    high.y += margin;
    constexpr float LargestFiniteFloat = 3.402823466e38f;
    float minX = LargestFiniteFloat, minY = LargestFiniteFloat, maxX = -LargestFiniteFloat,
          maxY = -LargestFiniteFloat;
    bool beforeNearPlane = false, visibleDepth = false;
    for (unsigned mask = 0; mask < 8; ++mask)
    {
        const float3 position =
            make_float3(mask & 1 ? high.x : low.x, mask & 2 ? high.y : low.y, mask & 4 ? high.z : low.z);
        const float3 delta = DicingSubtract(position, DicingVector(controls.CameraPositionTarget));
        const float z = DicingDot(delta, DicingVector(controls.CameraForwardHalfHeight));
        if (z <= DicingNearPlane)
        {
            beforeNearPlane = true;
            continue;
        }
        visibleDepth = true;
        const float x = DicingDot(delta, DicingVector(controls.CameraRightHalfWidth)) /
                        (z * controls.CameraRightHalfWidth.w);
        const float y = DicingDot(delta, DicingVector(controls.CameraUpAmplitude)) /
                        (z * controls.CameraForwardHalfHeight.w);
        minX = fminf(minX, x);
        minY = fminf(minY, y);
        maxX = fmaxf(maxX, x);
        maxY = fmaxf(maxY, y);
    }
    if (visibleDepth && beforeNearPlane)
        return controls.Extent.z | DicingClamped;
    const float pixelsX = !visibleDepth ? 0 : controls.Extent.x * (beforeNearPlane ? 1 : (maxX - minX) / 2);
    const float pixelsY = !visibleDepth ? 0 : controls.Extent.y * (beforeNearPlane ? 1 : (maxY - minY) / 2);
    float requested =
        fmaxf(1.f, sqrtf(pixelsX * pixelsX + pixelsY * pixelsY) / controls.CameraPositionTarget.w);
    if (displaced && margin > 0)
        requested = fmaxf(requested, DicingWaveSegments * (high.x - low.x) / controls.Wave.x);
    unsigned level = 0;
    for (; level < controls.Extent.z && float(1u << level) < requested; ++level)
    {
    }
    return level | (float(1u << level) < requested ? DicingClamped : 0);
}
__device__ float4 DicingBlend(float4 a, float4 b, float4 c, float3 w)
{
    return make_float4(a.x * w.x + b.x * w.y + c.x * w.z, a.y * w.x + b.y * w.y + c.y * w.z,
                       a.z * w.x + b.z * w.y + c.z * w.z, a.w * w.x + b.w * w.y + c.w * w.z);
}
__device__ DicingVertex DicingPoint(DicingPatch patch, unsigned i, unsigned j, unsigned n,
                                    DicingControls controls)
{
    const unsigned weights[]{n - i - j, i, j}, ids[]{patch.CornerIds.x, patch.CornerIds.y, patch.CornerIds.z};
    const float3 w = make_float3(float(weights[0]) / n, float(weights[1]) / n, float(weights[2]) / n);
    float3 position;
    unsigned zero = 3;
    for (unsigned corner = 0; corner < 3; ++corner)
        if (weights[corner] == 0)
        {
            zero = corner;
            break;
        }
    if (zero < 3)
    {
        unsigned a = (zero + 1) % 3, b = (zero + 2) % 3;
        if (ids[b] < ids[a])
        {
            const unsigned temporary = a;
            a = b;
            b = temporary;
        }
        const float3 first = DicingVector(patch.Corners[a].Position),
                     last = DicingVector(patch.Corners[b].Position);
        position = weights[a] == 0 ? last
                   : weights[b] == 0
                       ? first
                       : DicingAdd(first, DicingScale(DicingSubtract(last, first), float(weights[b]) / n));
    }
    else
    {
        const float3 a = DicingVector(patch.Corners[0].Position);
        position = DicingAdd(
            DicingAdd(a, DicingScale(DicingSubtract(DicingVector(patch.Corners[1].Position), a), w.y)),
            DicingScale(DicingSubtract(DicingVector(patch.Corners[2].Position), a), w.z));
    }
    float3 normal = (patch.Metadata.y & DicingSmooth)
                        ? DicingVector(DicingBlend(patch.Corners[0].Normal, patch.Corners[1].Normal,
                                                   patch.Corners[2].Normal, w))
                        : DicingCross(DicingSubtract(DicingVector(patch.Corners[1].Position),
                                                     DicingVector(patch.Corners[0].Position)),
                                      DicingSubtract(DicingVector(patch.Corners[2].Position),
                                                     DicingVector(patch.Corners[0].Position)));
    if ((patch.Metadata.y & DicingDisplace) && controls.CameraUpAmplitude.w != 0)
    {
        const float cycles = position.x / controls.Wave.x + controls.Wave.y;
        const float radians = (cycles - floorf(cycles)) * DicingTwoPi;
        const float derivative =
            controls.CameraUpAmplitude.w * (DicingTwoPi / controls.Wave.x) * cosf(radians);
        normal.x -= derivative * normal.y;
        position.y += controls.CameraUpAmplitude.w * sinf(radians);
    }
    const float square = DicingDot(normal, normal);
    normal = square > 0 ? DicingScale(normal, 1 / sqrtf(square)) : make_float3(0, 0, 1);
    DicingVertex result;
    result.Position = make_float4(position.x, position.y, position.z, 0);
    result.Normal = make_float4(normal.x, normal.y, normal.z, 0);
    result.Color = DicingBlend(patch.Corners[0].Color, patch.Corners[1].Color, patch.Corners[2].Color, w);
    result.TexCoord =
        DicingBlend(patch.Corners[0].TexCoord, patch.Corners[1].TexCoord, patch.Corners[2].TexCoord, w);
    return result;
}
__device__ DicingTriangle DicingGenerate(DicingPatch patch, unsigned ordinal, unsigned segments,
                                         DicingControls controls)
{
    unsigned row = 0, stop = segments;
    while (stop - row > 1)
    {
        const unsigned middle = row + (stop - row) / 2;
        const unsigned prefix = middle * (2 * segments - middle);
        if (prefix <= ordinal)
            row = middle;
        else
            stop = middle;
    }
    const unsigned remainder = ordinal - row * (2 * segments - row), column = remainder / 2;
    DicingTriangle result;
    result.Corners[0] = DicingPoint(patch, row + (remainder & 1), column, segments, controls);
    result.Corners[1] = DicingPoint(patch, row + 1, column + (remainder & 1), segments, controls);
    result.Corners[2] = DicingPoint(patch, row, column + 1, segments, controls);
    return result;
}
__device__ void DicingStore(unsigned* destination, float4 value)
{
    destination[0] = __float_as_uint(value.x);
    destination[1] = __float_as_uint(value.y);
    destination[2] = __float_as_uint(value.z);
    destination[3] = __float_as_uint(value.w);
}
extern "C" __global__ void DiceMicropolygons(DicingControls controls, const DicingPatch* patches,
                                             const uint4* ranges, unsigned* output)
{
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= controls.Dispatch.y)
        return;
    const unsigned index = controls.Dispatch.x + lane;
    if (controls.Populations.w == 0)
    {
        atomicMax(output + patches[index].Metadata.z, DicingPlan(patches[index], controls));
        return;
    }
    unsigned begin = 0, end = controls.Populations.x;
    while (begin < end)
    {
        const unsigned middle = begin + (end - begin) / 2;
        if (ranges[middle].y > index)
            end = middle;
        else
            begin = middle + 1;
    }
    const uint4 range = ranges[begin];
    const DicingTriangle triangle = DicingGenerate(patches[begin], index - range.x, range.z, controls);
    constexpr unsigned VertexWords = 16, VerticesPerTriangle = 3;
    for (unsigned vertex = 0; vertex < VerticesPerTriangle; ++vertex)
    {
        unsigned* destination = output + (index * VerticesPerTriangle + vertex) * VertexWords;
        DicingStore(destination, triangle.Corners[vertex].Position);
        DicingStore(destination + 4, triangle.Corners[vertex].Normal);
        DicingStore(destination + 8, triangle.Corners[vertex].Color);
        DicingStore(destination + 12, triangle.Corners[vertex].TexCoord);
    }
}
