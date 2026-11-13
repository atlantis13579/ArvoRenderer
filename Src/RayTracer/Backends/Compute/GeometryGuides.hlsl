// Independently authored Compute camera/geometry guides for reprojection.
// Guides preserve geometric orientation and full-film pixel identity, including
// crop halos. A pinhole probe is not an aperture-averaged visibility sample.
struct GuideRay
{
    float4 OriginMin, DirectionMax;
};
struct QueryHit
{
    float3 Normal;
    float Distance;
    float3 Color;
    uint Primitive;
    float2 Uv;
    uint Material, Flags;
    float3 GeometricNormal;
    uint Reserved;
};
struct GuidePixel
{
    float3 Position;
    float Depth;
    float3 Normal;
    float Distance;
    float3 Direction;
    uint Flags;
    uint Primitive, Material;
    uint2 Reserved;
};
StructuredBuffer<GuideRay> InputRays : register(t0);
StructuredBuffer<QueryHit> Hits : register(t1);
RWStructuredBuffer<GuideRay> GeneratedRays : register(u0);
RWStructuredBuffer<GuidePixel> Guides : register(u1);
RWStructuredBuffer<uint> Errors : register(u2);
cbuffer GuideControls : register(b0)
{
    uint Width, Height, FullWidth, FullHeight;
    uint TileX, TileY, Begin, Count;
    float3 CameraPosition;
    float Fov;
    float3 CameraTarget;
    float MinimumDistance;
    float3 CameraUp;
    float MaximumDistance;
    uint4 Reserved;
};
static const uint Threads = 64, MaximumExtent = 16384, BadConfiguration = 1, BadValue = 2;
static const uint Surface = 1, Background = 2, Front = 4, InvalidPrimitive = 0xffffffffu;
static const float Pi = 3.14159265358979323846f, HalfTurn = 180, Center = 0.5f;
float3 GuideUnit(float3 v)
{
    const float peak = max(abs(v.x), max(abs(v.y), abs(v.z)));
    if (!(peak > 0))
        return 0;
    v /= peak;
    return v / sqrt(dot(v, v));
}
bool ValidGuideWindow()
{
    return Width && Height && FullWidth && FullHeight && FullWidth <= MaximumExtent &&
           FullHeight <= MaximumExtent && TileX <= FullWidth && Width <= FullWidth - TileX &&
           TileY <= FullHeight && Height <= FullHeight - TileY && Begin <= Width * Height &&
           Count <= Width * Height - Begin;
}
[numthreads(Threads, 1, 1)] void GenerateGeometryGuideRays(uint3 dispatch : SV_DispatchThreadID)
{
    if (dispatch.x >= Count)
        return;
    if (!ValidGuideWindow())
    {
        InterlockedOr(Errors[0], BadConfiguration);
        return;
    }
    const uint index = Begin + dispatch.x;
    const float3 forward = GuideUnit(CameraTarget - CameraPosition);
    const float3 right = GuideUnit(cross(forward, CameraUp)), up = cross(right, forward);
    const float halfHeight = tan(Fov * Pi / (2 * HalfTurn));
    const float2 halfExtent = float2(halfHeight * float(FullWidth) / float(FullHeight), halfHeight);
    const float2 raster = float2(index % Width + TileX, index / Width + TileY) + Center;
    const float2 screen = float2(2 * raster.x / FullWidth - 1, 1 - 2 * raster.y / FullHeight) * halfExtent;
    const float3 direction = GuideUnit(forward + right * screen.x + up * screen.y);
    if (any(!isfinite(direction)) || !(dot(direction, direction) > 0))
    {
        InterlockedOr(Errors[0], BadValue);
        return;
    }
    GuideRay ray;
    ray.OriginMin = float4(CameraPosition, MinimumDistance);
    ray.DirectionMax = float4(direction, MaximumDistance);
    GeneratedRays[index] = ray;
}
[numthreads(Threads, 1, 1)] void ResolveGeometryGuideHits(uint3 dispatch : SV_DispatchThreadID)
{
    if (dispatch.x >= Count)
        return;
    if (!ValidGuideWindow())
    {
        InterlockedOr(Errors[0], BadConfiguration);
        return;
    }
    const uint index = Begin + dispatch.x;
    const GuideRay ray = InputRays[index];
    const QueryHit hit = Hits[index];
    GuidePixel guide = (GuidePixel)0;
    guide.Direction = ray.DirectionMax.xyz;
    guide.Flags = Background;
    guide.Primitive = InvalidPrimitive;
    if (hit.Flags & Surface)
    {
        const float3 forward = GuideUnit(CameraTarget - CameraPosition);
        guide.Position = float3(double3(ray.OriginMin.xyz) + double3(guide.Direction) * double(hit.Distance));
        guide.Normal = GuideUnit(hit.GeometricNormal);
        guide.Depth = hit.Distance * dot(guide.Direction, forward);
        guide.Distance = hit.Distance;
        guide.Primitive = hit.Primitive;
        guide.Material = hit.Material;
        guide.Flags = Surface | (hit.Flags & Front);
        if (any(!isfinite(guide.Position)) || any(!isfinite(guide.Normal)) ||
            !(dot(guide.Normal, guide.Normal) > 0) || !isfinite(guide.Depth) || !(guide.Depth > 0) ||
            !isfinite(hit.Distance) || hit.Distance <= ray.OriginMin.w ||
            hit.Distance >= ray.DirectionMax.w || hit.Primitive == InvalidPrimitive)
        {
            InterlockedOr(Errors[0], BadValue);
            return;
        }
    }
    Guides[index] = guide;
}
