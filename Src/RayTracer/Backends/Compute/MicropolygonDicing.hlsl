// Independent compute dicing and analytic displacement. A component-wide
// power-of-two lattice closes shared edges; canonical corner IDs fix their
// interpolation order. Generated triangles feed this backend's acceleration.
// Cook, Carpenter and Catmull, "The Reyes Image Rendering Architecture" (1987).
struct DiceVertex
{
    float4 Position, Normal, Color, TexCoord;
};
struct DicePatch
{
    DiceVertex Corners[3];
    uint4 Metadata, CornerIds;
};
struct DiceTriangle
{
    DiceVertex Corners[3];
};
struct DiceControls
{
    float4 CameraPositionTarget, CameraForwardHalfHeight, CameraRightHalfWidth, CameraUpAmplitude, Wave;
    uint4 Populations, Extent, Dispatch;
};
static const uint DiceSmooth = 1, DiceDisplace = 2, DiceClamped = 1u << 8;
static const float DicePi = 3.14159265358979323846f;
static const float DiceNear = 1.e-4f, DiceSegmentsPerWave = 8;
float3 DiceUnit(float3 normal)
{
    const float square = dot(normal, normal);
    return square > 0 ? normal / sqrt(square) : float3(0, 0, 1);
}
uint DicePlan(DicePatch patch, DiceControls controls)
{
    float3 lower = patch.Corners[0].Position.xyz, upper = lower;
    for (uint vertex = 1; vertex < 3; ++vertex)
    {
        lower = min(lower, patch.Corners[vertex].Position.xyz);
        upper = max(upper, patch.Corners[vertex].Position.xyz);
    }
    const bool displaced = (patch.Metadata.y & DiceDisplace) != 0;
    const float amplitude = displaced ? abs(controls.CameraUpAmplitude.w) : 0;
    lower.y -= amplitude;
    upper.y += amplitude;
    const float LargestFiniteFloat = 3.402823466e38f;
    float2 low = float2(LargestFiniteFloat, LargestFiniteFloat), high = -low;
    bool front = false, behind = false;
    for (uint corner = 0; corner < (1u << 3); ++corner)
    {
        const float3 p = float3(corner & 1 ? upper.x : lower.x, corner & 2 ? upper.y : lower.y,
                                corner & 4 ? upper.z : lower.z) -
                         controls.CameraPositionTarget.xyz;
        precise float z = dot(p, controls.CameraForwardHalfHeight.xyz);
        behind = behind || z <= DiceNear;
        if (z > DiceNear)
        {
            front = true;
            precise float2 projected =
                float2(dot(p, controls.CameraRightHalfWidth.xyz) / (z * controls.CameraRightHalfWidth.w),
                       dot(p, controls.CameraUpAmplitude.xyz) / (z * controls.CameraForwardHalfHeight.w));
            low = min(low, projected);
            high = max(high, projected);
        }
    }
    float2 pixels = float2(0, 0);
    if (front && behind)
        return controls.Extent.z | DiceClamped;
    if (front)
        pixels = float2(controls.Extent.x, controls.Extent.y) * (behind ? float2(1, 1) : (high - low) / 2);
    float segments = max(1.0f, length(pixels) / controls.CameraPositionTarget.w);
    if (displaced && amplitude > 0)
        segments = max(segments, DiceSegmentsPerWave * (upper.x - lower.x) / controls.Wave.x);
    uint level = 0;
    while (level < controls.Extent.z && float(1u << level) < segments)
        ++level;
    return level | (float(1u << level) < segments ? DiceClamped : 0);
}
DiceVertex DiceInterpolate(DicePatch patch, uint2 gridPoint, uint segments, DiceControls controls)
{
    const uint3 weights = uint3(segments - gridPoint.x - gridPoint.y, gridPoint.x, gridPoint.y);
    const float3 barycentric = float3(weights) / float(segments);
    precise float3 position = float3(0, 0, 0);
    bool edge = false;
    for (uint zero = 0; zero < 3; ++zero)
        if (weights[zero] == 0)
        {
            uint first = (zero + 1) % 3, second = (zero + 2) % 3;
            if (patch.CornerIds[first] > patch.CornerIds[second])
            {
                uint swap = first;
                first = second;
                second = swap;
            }
            const float3 a = patch.Corners[first].Position.xyz, b = patch.Corners[second].Position.xyz;
            position = weights[first] == 0    ? b
                       : weights[second] == 0 ? a
                                              : a + (b - a) * (float(weights[second]) / segments);
            edge = true;
            break;
        }
    if (!edge)
        position = patch.Corners[0].Position.xyz +
                   (patch.Corners[1].Position.xyz - patch.Corners[0].Position.xyz) * barycentric.y +
                   (patch.Corners[2].Position.xyz - patch.Corners[0].Position.xyz) * barycentric.z;
    float3 normal = (patch.Metadata.y & DiceSmooth) != 0
                        ? patch.Corners[0].Normal.xyz * barycentric.x +
                              patch.Corners[1].Normal.xyz * barycentric.y +
                              patch.Corners[2].Normal.xyz * barycentric.z
                        : cross(patch.Corners[1].Position.xyz - patch.Corners[0].Position.xyz,
                                patch.Corners[2].Position.xyz - patch.Corners[0].Position.xyz);
    if ((patch.Metadata.y & DiceDisplace) != 0 && controls.CameraUpAmplitude.w != 0)
    {
        const float cycles = position.x / controls.Wave.x + controls.Wave.y;
        const float angle = (cycles - floor(cycles)) * (2 * DicePi);
        const float derivative = controls.CameraUpAmplitude.w * (2 * DicePi / controls.Wave.x) * cos(angle);
        position.y += controls.CameraUpAmplitude.w * sin(angle);
        normal.x -= derivative * normal.y;
    }
    DiceVertex vertex;
    vertex.Position = float4(position, 0);
    vertex.Normal = float4(DiceUnit(normal), 0);
    vertex.Color = patch.Corners[0].Color * barycentric.x + patch.Corners[1].Color * barycentric.y +
                   patch.Corners[2].Color * barycentric.z;
    vertex.TexCoord = patch.Corners[0].TexCoord * barycentric.x + patch.Corners[1].TexCoord * barycentric.y +
                      patch.Corners[2].TexCoord * barycentric.z;
    return vertex;
}
DiceTriangle DiceGenerate(DicePatch patch, uint local, uint segments, DiceControls controls)
{
    uint lower = 0, upper = segments;
    while (upper - lower > 1)
    {
        const uint middle = (lower + upper) / 2;
        if (middle * (2 * segments - middle) <= local)
            lower = middle;
        else
            upper = middle;
    }
    const uint ordinal = local - lower * (2 * segments - lower), column = ordinal / 2;
    const uint2 first = uint2(lower + (ordinal & 1), column);
    const uint2 second = (ordinal & 1) == 0 ? uint2(lower + 1, column) : uint2(lower + 1, column + 1);
    DiceTriangle output;
    output.Corners[0] = DiceInterpolate(patch, first, segments, controls);
    output.Corners[1] = DiceInterpolate(patch, second, segments, controls);
    output.Corners[2] = DiceInterpolate(patch, uint2(lower, column + 1), segments, controls);
    return output;
}
StructuredBuffer<DicePatch> Patches : register(t0);
StructuredBuffer<uint4> Ranges : register(t1);
RWStructuredBuffer<uint> OutputWords : register(u0);
cbuffer DicingSettings : register(b0)
{
    DiceControls Controls;
};
void DiceStore(uint offset, float4 value)
{
    const uint4 bits = asuint(value);
    OutputWords[offset] = bits.x;
    OutputWords[offset + 1] = bits.y;
    OutputWords[offset + 2] = bits.z;
    OutputWords[offset + 3] = bits.w;
}
[RootSignature("RootConstants(num32BitConstants=32,b0),DescriptorTable(SRV(t0,numDescriptors=2)),"
               "DescriptorTable(UAV(u0))")][numthreads(64, 1, 1)] void
DiceMicropolygons(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= Controls.Dispatch.y)
        return;
    const uint index = Controls.Dispatch.x + thread.x;
    if (Controls.Populations.w == 0)
    {
        InterlockedMax(OutputWords[Patches[index].Metadata.z], DicePlan(Patches[index], Controls));
        return;
    }
    uint begin = 0, end = Controls.Populations.x;
    while (begin < end)
    {
        const uint middle = (begin + end) / 2;
        if (Ranges[middle].y <= index)
            begin = middle + 1;
        else
            end = middle;
    }
    const uint4 range = Ranges[begin];
    const DiceTriangle generated = DiceGenerate(Patches[begin], index - range.x, range.z, Controls);
    const uint WordsPerVertex = 16, VerticesPerTriangle = 3;
    for (uint corner = 0; corner < VerticesPerTriangle; ++corner)
    {
        const uint offset = (index * VerticesPerTriangle + corner) * WordsPerVertex;
        DiceStore(offset, generated.Corners[corner].Position);
        DiceStore(offset + 4, generated.Corners[corner].Normal);
        DiceStore(offset + 8, generated.Corners[corner].Color);
        DiceStore(offset + 12, generated.Corners[corner].TexCoord);
    }
}
