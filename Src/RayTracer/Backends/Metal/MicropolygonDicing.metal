// Native Metal component dicing, continuous analytic displacement, and the
// corresponding normal Jacobian. This is geometry generation before native
// acceleration construction, not a bump-map approximation.
// Cook, Carpenter and Catmull, "The Reyes Image Rendering Architecture" (1987).
#include <metal_stdlib>
using namespace metal;
struct MicropolygonVertex
{
    float4 Position, Normal, Color, TexCoord;
};
struct MicropolygonPatch
{
    MicropolygonVertex Corners[3];
    uint4 Metadata, CornerIds;
};
struct MicropolygonTriangle
{
    MicropolygonVertex Corners[3];
};
struct MicropolygonControls
{
    float4 CameraPositionTarget, CameraForwardHalfHeight, CameraRightHalfWidth, CameraUpAmplitude, Wave;
    uint4 Populations, Extent, Dispatch;
};
constant float MicropolygonNear = 1.e-4f, MicropolygonWaveSegments = 8;
constant float MicropolygonTau = 2 * 3.14159265358979323846f;
constant uint MicropolygonSmooth = 1, MicropolygonWave = 2, MicropolygonClamped = 1u << 8;
uint MicropolygonLevel(MicropolygonPatch patch, MicropolygonControls controls)
{
    float3 minimum = patch.Corners[0].Position.xyz, maximum = minimum;
    for (uint corner = 1; corner != 3; ++corner)
    {
        minimum = min(minimum, patch.Corners[corner].Position.xyz);
        maximum = max(maximum, patch.Corners[corner].Position.xyz);
    }
    const bool wave = (patch.Metadata.y & MicropolygonWave) != 0;
    const float amplitude = wave ? abs(controls.CameraUpAmplitude.w) : 0;
    minimum.y -= amplitude;
    maximum.y += amplitude;
    const float LargestFiniteFloat = 3.402823466e38f;
    float2 low = float2(LargestFiniteFloat), high = -low;
    bool crossing = false, inFront = false;
    for (uint bits = 0; bits < (1u << 3); ++bits)
    {
        const float3 corner = float3((bits & 1) ? maximum.x : minimum.x, (bits & 2) ? maximum.y : minimum.y,
                                     (bits & 4) ? maximum.z : minimum.z);
        const float3 offset = corner - controls.CameraPositionTarget.xyz;
        const float z = dot(offset, controls.CameraForwardHalfHeight.xyz);
        crossing |= z <= MicropolygonNear;
        if (z <= MicropolygonNear)
            continue;
        inFront = true;
        const float2 projected =
            float2(dot(offset, controls.CameraRightHalfWidth.xyz) / (z * controls.CameraRightHalfWidth.w),
                   dot(offset, controls.CameraUpAmplitude.xyz) / (z * controls.CameraForwardHalfHeight.w));
        low = min(low, projected);
        high = max(high, projected);
    }
    if (inFront && crossing)
        return controls.Extent.z | MicropolygonClamped;
    const float2 span =
        inFront ? float2(controls.Extent.x, controls.Extent.y) * (crossing ? float2(1) : (high - low) / 2)
                : float2(0);
    float demand = max(1.0f, sqrt(dot(span, span)) / controls.CameraPositionTarget.w);
    if (wave && amplitude > 0)
        demand = max(demand, MicropolygonWaveSegments * (maximum.x - minimum.x) / controls.Wave.x);
    uint level = 0;
    while (level < controls.Extent.z)
    {
        if (float(1u << level) >= demand)
            break;
        ++level;
    }
    return level | (float(1u << level) < demand ? MicropolygonClamped : 0);
}
MicropolygonVertex MicropolygonPoint(MicropolygonPatch patch, uint row, uint column, uint segments,
                                     MicropolygonControls controls)
{
    const uint3 weights = uint3(segments - row - column, row, column);
    const float3 fractions = float3(weights) / segments;
    float3 position;
    if (all(weights != uint3(0)))
    {
        const float3 origin = patch.Corners[0].Position.xyz;
        position = origin + (patch.Corners[1].Position.xyz - origin) * fractions.y +
                   (patch.Corners[2].Position.xyz - origin) * fractions.z;
    }
    else
    {
        const uint edge = weights.x == 0 ? 0 : weights.y == 0 ? 1 : 2;
        uint first = (edge + 1) % 3, last = (edge + 2) % 3;
        if (patch.CornerIds[last] < patch.CornerIds[first])
        {
            const uint saved = first;
            first = last;
            last = saved;
        }
        const float3 a = patch.Corners[first].Position.xyz, b = patch.Corners[last].Position.xyz;
        position = weights[first] == 0  ? b
                   : weights[last] == 0 ? a
                                        : a + (b - a) * (float(weights[last]) / segments);
    }
    float3 normal;
    if ((patch.Metadata.y & MicropolygonSmooth) == 0)
        normal = cross(patch.Corners[1].Position.xyz - patch.Corners[0].Position.xyz,
                       patch.Corners[2].Position.xyz - patch.Corners[0].Position.xyz);
    else
        normal = patch.Corners[0].Normal.xyz * fractions.x + patch.Corners[1].Normal.xyz * fractions.y +
                 patch.Corners[2].Normal.xyz * fractions.z;
    if ((patch.Metadata.y & MicropolygonWave) != 0 && controls.CameraUpAmplitude.w != 0)
    {
        const float coordinate = position.x / controls.Wave.x + controls.Wave.y;
        const float angle = MicropolygonTau * (coordinate - floor(coordinate));
        const float slope = controls.CameraUpAmplitude.w * (MicropolygonTau / controls.Wave.x) * cos(angle);
        normal.x -= slope * normal.y;
        position.y += controls.CameraUpAmplitude.w * sin(angle);
    }
    const float norm = dot(normal, normal);
    MicropolygonVertex outVertex;
    outVertex.Position = float4(position, 0);
    outVertex.Normal = float4(norm > 0 ? normal / sqrt(norm) : float3(0, 0, 1), 0);
    outVertex.Color = patch.Corners[0].Color * fractions.x + patch.Corners[1].Color * fractions.y +
                      patch.Corners[2].Color * fractions.z;
    outVertex.TexCoord = patch.Corners[0].TexCoord * fractions.x + patch.Corners[1].TexCoord * fractions.y +
                        patch.Corners[2].TexCoord * fractions.z;
    return outVertex;
}
MicropolygonTriangle MicropolygonGenerate(MicropolygonPatch patch, uint ordinal, uint n,
                                          MicropolygonControls controls)
{
    uint row = n - uint(ceil(sqrt(float(n * n - ordinal))));
    while (row != 0 && row * (2 * n - row) > ordinal)
        --row;
    while (row + 1 < n && (row + 1) * (2 * n - row - 1) <= ordinal)
        ++row;
    const uint inRow = ordinal - row * (2 * n - row), column = inRow / 2, inverted = inRow & 1;
    MicropolygonTriangle triangle;
    triangle.Corners[0] = MicropolygonPoint(patch, row + inverted, column, n, controls);
    triangle.Corners[1] = MicropolygonPoint(patch, row + 1, column + inverted, n, controls);
    triangle.Corners[2] = MicropolygonPoint(patch, row, column + 1, n, controls);
    return triangle;
}
void MicropolygonWrite(device uint* words, float4 value)
{
    const uint4 bits = as_type<uint4>(value);
    words[0] = bits.x;
    words[1] = bits.y;
    words[2] = bits.z;
    words[3] = bits.w;
}
kernel void ArvoDiceMicropolygons(constant MicropolygonControls& controls [[buffer(0)]],
                                  device const MicropolygonPatch* patches [[buffer(1)]],
                                  device const uint4* ranges [[buffer(2)]], device uint* output [[buffer(3)]],
                                  uint lane [[thread_position_in_grid]])
{
    if (lane >= controls.Dispatch.y)
        return;
    const uint index = controls.Dispatch.x + lane;
    if (controls.Populations.w == 0)
    {
        device atomic_uint* levels = reinterpret_cast<device atomic_uint*>(output);
        atomic_fetch_max_explicit(levels + patches[index].Metadata.z,
                                  MicropolygonLevel(patches[index], controls), memory_order_relaxed);
        return;
    }
    uint lo = 0, hi = controls.Populations.x;
    while (lo < hi)
    {
        const uint mid = lo + (hi - lo) / 2;
        if (ranges[mid].y <= index)
            lo = mid + 1;
        else
            hi = mid;
    }
    const uint4 range = ranges[lo];
    const MicropolygonTriangle triangle =
        MicropolygonGenerate(patches[lo], index - range.x, range.z, controls);
    const uint WordsPerVertex = 16, VerticesPerTriangle = 3;
    for (uint vertexData = 0; vertexData < VerticesPerTriangle; ++vertexData)
    {
        device uint* words = output + (index * VerticesPerTriangle + vertexData) * WordsPerVertex;
        MicropolygonWrite(words, triangle.Corners[vertexData].Position);
        MicropolygonWrite(words + 4, triangle.Corners[vertexData].Normal);
        MicropolygonWrite(words + 8, triangle.Corners[vertexData].Color);
        MicropolygonWrite(words + 12, triangle.Corners[vertexData].TexCoord);
    }
}
