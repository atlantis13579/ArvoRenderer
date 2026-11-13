// Independent DX11 preview geometry program. Cook et al., "The Reyes Image
// Rendering Architecture" (1987), https://graphics.pixar.com/library/Reyes/
// Dicing happens before raster visibility; this is not an offline integrator.
// One power-of-two level per connected component preserves shared edges.
struct PreviewVertex
{
    float4 Position, Normal, Color, TexCoord;
};
struct PreviewPatch
{
    PreviewVertex Corners[3];
    uint4 Metadata, CornerIds;
};
struct PreviewTriangle
{
    PreviewVertex Corners[3];
};
struct PreviewDicingControls
{
    float4 CameraPositionTarget, CameraForwardHalfHeight, CameraRightHalfWidth, CameraUpAmplitude, Wave;
    uint4 Populations, Extent, Dispatch;
};
static const uint PreviewSmoothNormal = 1, PreviewDisplacement = 2, PreviewLevelLimited = 1u << 8;
static const float PreviewPi = 3.14159265358979323846f;
static const float PreviewProjectionNear = 1.e-4f, PreviewSamplesPerPeriod = 8;
float3 PreviewUnit(float3 value)
{
    const float square = dot(value, value);
    return square > 0 ? value / sqrt(square) : float3(0, 0, 1);
}
uint PreviewLevel(PreviewPatch patch, PreviewDicingControls frame)
{
    float3 minimum = patch.Corners[0].Position.xyz, maximum = minimum;
    for (uint corner = 1; corner < 3; ++corner)
    {
        minimum = min(minimum, patch.Corners[corner].Position.xyz);
        maximum = max(maximum, patch.Corners[corner].Position.xyz);
    }
    const bool wave = (patch.Metadata.y & PreviewDisplacement) != 0;
    const float margin = wave ? abs(frame.CameraUpAmplitude.w) : 0;
    minimum.y -= margin;
    maximum.y += margin;
    const float LargestFiniteFloat = 3.402823466e38f;
    float2 screenMinimum = float2(LargestFiniteFloat, LargestFiniteFloat), screenMaximum = -screenMinimum;
    uint visible = 0, clipped = 0;
    for (uint boxCorner = 0; boxCorner < (1u << 3); ++boxCorner)
    {
        const float3 position =
            float3((boxCorner & 1) ? maximum.x : minimum.x, (boxCorner & 2) ? maximum.y : minimum.y,
                   (boxCorner & 4) ? maximum.z : minimum.z);
        const float3 relative = position - frame.CameraPositionTarget.xyz;
        const float depth = dot(relative, frame.CameraForwardHalfHeight.xyz);
        if (depth <= PreviewProjectionNear)
        {
            ++clipped;
            continue;
        }
        ++visible;
        const float2 projected =
            float2(dot(relative, frame.CameraRightHalfWidth.xyz) / (depth * frame.CameraRightHalfWidth.w),
                   dot(relative, frame.CameraUpAmplitude.xyz) / (depth * frame.CameraForwardHalfHeight.w));
        screenMinimum = min(screenMinimum, projected);
        screenMaximum = max(screenMaximum, projected);
    }
    if (visible != 0 && clipped != 0)
        return frame.Extent.z | PreviewLevelLimited;
    // Use the whole projected patch, including offscreen portions of its domain.
    const float2 span =
        visible ? (screenMaximum - screenMinimum) * float2(frame.Extent.x, frame.Extent.y) / 2 : float2(0, 0);
    float density = max(1.0f, length(span) / frame.CameraPositionTarget.w);
    if (wave && margin > 0)
        density = max(density, (maximum.x - minimum.x) * PreviewSamplesPerPeriod / frame.Wave.x);
    uint level = 0;
    for (; level < frame.Extent.z && float(1u << level) < density; ++level)
    {
    }
    return level | (float(1u << level) < density ? PreviewLevelLimited : 0);
}
PreviewVertex PreviewGridPoint(PreviewPatch patch, uint row, uint column, uint segments,
                               PreviewDicingControls frame)
{
    const uint3 weights = uint3(segments - row - column, row, column);
    const float3 barycentric = float3(weights) / float(segments);
    precise float3 position = patch.Corners[0].Position.xyz;
    bool sharedEdge = false;
    for (uint corner = 0; corner < 3; ++corner)
    {
        if (weights[corner] != 0)
            continue;
        uint first = (corner + 1) % 3, second = (corner + 2) % 3;
        if (patch.CornerIds[first] > patch.CornerIds[second])
        {
            const uint temporary = first;
            first = second;
            second = temporary;
        }
        const float3 a = patch.Corners[first].Position.xyz, b = patch.Corners[second].Position.xyz;
        if (weights[first] == segments)
            position = a;
        else if (weights[second] == segments)
            position = b;
        else
            position = a + (b - a) * (float(weights[second]) / segments);
        sharedEdge = true;
        break;
    }
    if (!sharedEdge)
        position = patch.Corners[0].Position.xyz +
                   (patch.Corners[1].Position.xyz - patch.Corners[0].Position.xyz) * barycentric.y +
                   (patch.Corners[2].Position.xyz - patch.Corners[0].Position.xyz) * barycentric.z;
    float3 normal = (patch.Metadata.y & PreviewSmoothNormal)
                        ? patch.Corners[0].Normal.xyz * barycentric.x +
                              patch.Corners[1].Normal.xyz * barycentric.y +
                              patch.Corners[2].Normal.xyz * barycentric.z
                        : cross(patch.Corners[1].Position.xyz - patch.Corners[0].Position.xyz,
                                patch.Corners[2].Position.xyz - patch.Corners[0].Position.xyz);
    if ((patch.Metadata.y & PreviewDisplacement) != 0)
    {
        const float angle = (2 * PreviewPi) * frac(position.x / frame.Wave.x + frame.Wave.y);
        position.y += frame.CameraUpAmplitude.w * sin(angle);
        const float slope = frame.CameraUpAmplitude.w * (2 * PreviewPi) / frame.Wave.x * cos(angle);
        normal.x -= slope * normal.y;
    }
    PreviewVertex result;
    result.Position = float4(position, 0);
    result.Normal = float4(PreviewUnit(normal), 0);
    result.Color = patch.Corners[0].Color * barycentric.x + patch.Corners[1].Color * barycentric.y +
                   patch.Corners[2].Color * barycentric.z;
    result.TexCoord = patch.Corners[0].TexCoord * barycentric.x + patch.Corners[1].TexCoord * barycentric.y +
                      patch.Corners[2].TexCoord * barycentric.z;
    return result;
}
PreviewTriangle PreviewGridTriangle(PreviewPatch patch, uint ordinal, uint segments,
                                    PreviewDicingControls frame)
{
    uint low = 0, high = segments;
    while (low + 1 < high)
    {
        const uint middle = low + (high - low) / 2;
        if (middle * (2 * segments - middle) <= ordinal)
            low = middle;
        else
            high = middle;
    }
    const uint withinRow = ordinal - low * (2 * segments - low), column = withinRow / 2;
    PreviewTriangle result;
    const uint alternating = withinRow & 1;
    result.Corners[0] = PreviewGridPoint(patch, low + alternating, column, segments, frame);
    result.Corners[1] = PreviewGridPoint(patch, low + 1, column + alternating, segments, frame);
    result.Corners[2] = PreviewGridPoint(patch, low, column + 1, segments, frame);
    return result;
}
StructuredBuffer<PreviewPatch> PreviewPatches : register(t0);
StructuredBuffer<uint4> PreviewRanges : register(t1);
RWStructuredBuffer<uint> PreviewOutput : register(u0);
cbuffer PreviewSettings : register(b0)
{
    PreviewDicingControls PreviewFrame;
};
void PreviewWrite(uint offset, float4 value)
{
    const uint4 bits = asuint(value);
    for (uint channel = 0; channel < 4; ++channel)
        PreviewOutput[offset + channel] = bits[channel];
}
[numthreads(64, 1, 1)] void DicePreview(uint3 invocation : SV_DispatchThreadID) {
    if (invocation.x >= PreviewFrame.Dispatch.y)
        return;
    const uint index = PreviewFrame.Dispatch.x + invocation.x;
    if (PreviewFrame.Populations.w == 0)
    {
        const PreviewPatch patch = PreviewPatches[index];
        InterlockedMax(PreviewOutput[patch.Metadata.z], PreviewLevel(patch, PreviewFrame));
        return;
    }
    uint lower = 0, upper = PreviewFrame.Populations.x;
    while (lower < upper)
    {
        const uint middle = lower + (upper - lower) / 2;
        if (index < PreviewRanges[middle].y)
            upper = middle;
        else
            lower = middle + 1;
    }
    const uint4 range = PreviewRanges[lower];
    const PreviewTriangle generated =
        PreviewGridTriangle(PreviewPatches[lower], index - range.x, range.z, PreviewFrame);
    const uint WordsPerCorner = 16, CornersPerTriangle = 3;
    for (uint corner = 0; corner < CornersPerTriangle; ++corner)
    {
        const uint offset = (index * CornersPerTriangle + corner) * WordsPerCorner;
        PreviewWrite(offset, generated.Corners[corner].Position);
        PreviewWrite(offset + 4, generated.Corners[corner].Normal);
        PreviewWrite(offset + 8, generated.Corners[corner].Color);
        PreviewWrite(offset + 12, generated.Corners[corner].TexCoord);
    }
}
