// DXR-native micromesh preparation for offline rendering; preview skips dicing.
// Dicing precedes BLAS/TLAS construction. This independently authored raygen
// pass generates actual triangles, rather than changing only shading normals.
// Cook et al., "The Reyes Image Rendering Architecture" (SIGGRAPH 1987).
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
static const float DicingNear = .0001f, DicingTwoPi = 2 * 3.14159265358979323846f;
static const float DicingWaveResolution = 8;
static const uint DicingSmooth = 1, DicingWave = 2, DicingLimitFlag = 1u << 8;
uint DicingLevel(DicingPatch patch, DicingControls c)
{
    float3 minimum =
        min(patch.Corners[0].Position.xyz, min(patch.Corners[1].Position.xyz, patch.Corners[2].Position.xyz));
    float3 maximum =
        max(patch.Corners[0].Position.xyz, max(patch.Corners[1].Position.xyz, patch.Corners[2].Position.xyz));
    const bool wave = (patch.Metadata.y & DicingWave) != 0;
    const float radius = wave ? abs(c.CameraUpAmplitude.w) : 0;
    minimum.y -= radius;
    maximum.y += radius;
    const float LargestFiniteFloat = 3.402823466e38f;
    float2 boundsMin = float2(LargestFiniteFloat, LargestFiniteFloat), boundsMax = -boundsMin;
    uint positive = 0, clipped = 0;
    for (uint corner = 0; corner < 8; ++corner)
    {
        const float3 world =
            float3((corner & 1) ? maximum.x : minimum.x, (corner & 2) ? maximum.y : minimum.y,
                   (corner & 4) ? maximum.z : minimum.z);
        const float3 relative = world - c.CameraPositionTarget.xyz;
        precise float depth = dot(relative, c.CameraForwardHalfHeight.xyz);
        if (depth <= DicingNear)
        {
            ++clipped;
            continue;
        }
        ++positive;
        precise float x = dot(relative, c.CameraRightHalfWidth.xyz) / (depth * c.CameraRightHalfWidth.w);
        precise float y = dot(relative, c.CameraUpAmplitude.xyz) / (depth * c.CameraForwardHalfHeight.w);
        const float2 projected = float2(x, y);
        boundsMin = min(boundsMin, projected);
        boundsMax = max(boundsMax, projected);
    }
    if (positive != 0 && clipped != 0)
        return c.Extent.z | DicingLimitFlag;
    const float2 extent = positive == 0 ? float2(0, 0)
                                        : float2(c.Extent.x, c.Extent.y) *
                                              (clipped != 0 ? float2(1, 1) : (boundsMax - boundsMin) / 2);
    const float pixels = sqrt(dot(extent, extent));
    float requested = max(1.0f, pixels / c.CameraPositionTarget.w);
    if (wave && radius > 0)
        requested = max(requested, (maximum.x - minimum.x) * DicingWaveResolution / c.Wave.x);
    uint subdivisions = 1, level = 0;
    while (float(subdivisions) < requested && level < c.Extent.z)
    {
        subdivisions *= 2;
        ++level;
    }
    return level | (float(subdivisions) < requested ? DicingLimitFlag : 0);
}
DicingVertex DicingPoint(DicingPatch patch, uint u, uint v, uint n, DicingControls c)
{
    const uint weight[3] = {n - u - v, u, v};
    const float3 barycentric = float3(weight[0], weight[1], weight[2]) / n;
    precise float3 position = patch.Corners[0].Position.xyz;
    if (weight[0] != 0 && weight[1] != 0 && weight[2] != 0)
        position += (patch.Corners[1].Position.xyz - position) * barycentric.y +
                    (patch.Corners[2].Position.xyz - position) * barycentric.z;
    else
    {
        const uint zero = weight[0] == 0 ? 0 : weight[1] == 0 ? 1 : 2;
        uint a = (zero + 1) % 3, b = (zero + 2) % 3;
        if (patch.CornerIds[a] > patch.CornerIds[b])
        {
            const uint temporary = a;
            a = b;
            b = temporary;
        }
        const float3 first = patch.Corners[a].Position.xyz, last = patch.Corners[b].Position.xyz;
        position = weight[a] == 0   ? last
                   : weight[b] == 0 ? first
                                    : first + (last - first) * (float(weight[b]) / n);
    }
    float3 normal;
    if ((patch.Metadata.y & DicingSmooth) != 0)
        normal = patch.Corners[0].Normal.xyz * barycentric.x + patch.Corners[1].Normal.xyz * barycentric.y +
                 patch.Corners[2].Normal.xyz * barycentric.z;
    else
        normal = cross(patch.Corners[1].Position.xyz - patch.Corners[0].Position.xyz,
                       patch.Corners[2].Position.xyz - patch.Corners[0].Position.xyz);
    if ((patch.Metadata.y & DicingWave) != 0 && c.CameraUpAmplitude.w != 0)
    {
        const float periodCoordinate = position.x / c.Wave.x + c.Wave.y;
        const float angle = DicingTwoPi * (periodCoordinate - floor(periodCoordinate));
        const float gradient = c.CameraUpAmplitude.w * (DicingTwoPi / c.Wave.x) * cos(angle);
        normal.x -= gradient * normal.y;
        position.y += c.CameraUpAmplitude.w * sin(angle);
    }
    const float normalSquared = dot(normal, normal);
    DicingVertex result;
    result.Position = float4(position, 0);
    result.Normal = float4(normalSquared > 0 ? normal / sqrt(normalSquared) : float3(0, 0, 1), 0);
    result.Color = patch.Corners[0].Color * barycentric.x + patch.Corners[1].Color * barycentric.y +
                   patch.Corners[2].Color * barycentric.z;
    result.TexCoord = patch.Corners[0].TexCoord * barycentric.x + patch.Corners[1].TexCoord * barycentric.y +
                      patch.Corners[2].TexCoord * barycentric.z;
    return result;
}
DicingTriangle DicingGenerate(DicingPatch patch, uint ordinal, uint n, DicingControls controls)
{
    // Row r begins at n^2-(n-r)^2. Correct the square-root estimate with
    // integer comparisons so exact row boundaries do not depend on sqrt ulps.
    uint row = n - uint(ceil(sqrt(float(n * n - ordinal))));
    while (row > 0 && row * (2 * n - row) > ordinal)
        --row;
    while (row + 1 < n && (row + 1) * (2 * n - row - 1) <= ordinal)
        ++row;
    const uint residual = ordinal - row * (2 * n - row), column = residual / 2;
    DicingTriangle result;
    if ((residual & 1) != 0)
    {
        result.Corners[0] = DicingPoint(patch, row + 1, column, n, controls);
        result.Corners[1] = DicingPoint(patch, row + 1, column + 1, n, controls);
    }
    else
    {
        result.Corners[0] = DicingPoint(patch, row, column, n, controls);
        result.Corners[1] = DicingPoint(patch, row + 1, column, n, controls);
    }
    result.Corners[2] = DicingPoint(patch, row, column + 1, n, controls);
    return result;
}
StructuredBuffer<DicingPatch> Patches : register(t0);
StructuredBuffer<uint4> Ranges : register(t1);
RWStructuredBuffer<uint> OutputWords : register(u0);
cbuffer MicropolygonSettings : register(b0)
{
    DicingControls Controls;
};
void DicingWrite(uint offset, float4 value)
{
    const uint4 packed = asuint(value);
    for (uint channel = 0; channel < 4; ++channel)
        OutputWords[offset + channel] = packed[channel];
}
[shader("raygeneration")] void MicropolygonRayGeneration() {
    const uint index = Controls.Dispatch.x + DispatchRaysIndex().x;
    if (Controls.Populations.w == 0)
    {
        const DicingPatch patch = Patches[index];
        InterlockedMax(OutputWords[patch.Metadata.z], DicingLevel(patch, Controls));
        return;
    }
    uint lo = 0, hi = Controls.Populations.x;
    while (lo < hi)
    {
        const uint middle = lo + (hi - lo) / 2;
        if (index < Ranges[middle].y)
            hi = middle;
        else
            lo = middle + 1;
    }
    const uint4 range = Ranges[lo];
    const DicingTriangle generated = DicingGenerate(Patches[lo], index - range.x, range.z, Controls);
    const uint VertexWords = 16, Corners = 3;
    for (uint vertex = 0; vertex < Corners; ++vertex)
    {
        const uint address = (index * Corners + vertex) * VertexWords;
        DicingWrite(address, generated.Corners[vertex].Position);
        DicingWrite(address + 4, generated.Corners[vertex].Normal);
        DicingWrite(address + 8, generated.Corners[vertex].Color);
        DicingWrite(address + 12, generated.Corners[vertex].TexCoord);
    }
}
