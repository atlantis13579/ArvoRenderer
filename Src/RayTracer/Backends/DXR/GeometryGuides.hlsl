// DXR's independent geometry-guide library supports offline reconstruction.
// Ray-generation entries perform camera/resolve work; scene traversal
// uses this backend's native ray-query pipeline between these two dispatches.
struct EncodedGuideRay
{
    float4 EyeMinimum, DirectionMaximum;
};
struct NativeGuideHit
{
    float3 ShadingNormal;
    float Distance;
    float3 Color;
    uint Primitive;
    float2 Uv;
    uint Material, Flags;
    float3 WindingNormal;
    uint Padding;
};
struct GeometryGuide
{
    float3 Point;
    float CameraDepth;
    float3 Normal;
    float Distance;
    float3 Direction;
    uint Flags;
    uint Primitive, Material;
    uint2 Padding;
};
StructuredBuffer<EncodedGuideRay> Rays : register(t0);
StructuredBuffer<NativeGuideHit> Intersections : register(t1);
RWStructuredBuffer<EncodedGuideRay> CameraRays : register(u0);
RWStructuredBuffer<GeometryGuide> Result : register(u1);
RWStructuredBuffer<uint> Failure : register(u2);
cbuffer GeometryGuideControls : register(b0)
{
    uint TileWidth, TileHeight, ImageWidth, ImageHeight;
    uint Left, Top, First, Population;
    float3 Eye;
    float FieldOfView;
    float3 Target;
    float NearDistance;
    float3 Up;
    float FarDistance;
    uint4 Padding;
};
static const uint ExtentLimit = 16384, InvalidDispatch = 1, InvalidValue = 2;
static const uint SurfaceFlag = 1, SkyFlag = 2, FrontFlag = 4, NoPrimitive = 0xffffffffu;
static const float Pi = 3.14159265358979323846f, DegreesPerHalfTurn = 180;
float3 NormalizedGuide(float3 v)
{
    float magnitude = max(max(abs(v.x), abs(v.y)), abs(v.z));
    if (magnitude == 0)
        return 0;
    float3 scaled = v / magnitude;
    return scaled / sqrt(scaled.x * scaled.x + scaled.y * scaled.y + scaled.z * scaled.z);
}
bool GuideDispatchValid()
{
    if (!TileWidth || !TileHeight || !ImageWidth || !ImageHeight || ImageWidth > ExtentLimit ||
        ImageHeight > ExtentLimit)
        return false;
    if (Left > ImageWidth || Top > ImageHeight || TileWidth > ImageWidth - Left ||
        TileHeight > ImageHeight - Top)
        return false;
    const uint available = TileWidth * TileHeight;
    return First <= available && Population <= available - First;
}
[shader("raygeneration")] void GenerateGeometryGuideRays()
{
    const uint lane = DispatchRaysIndex().x;
    if (lane >= Population)
        return;
    if (!GuideDispatchValid())
    {
        InterlockedOr(Failure[0], InvalidDispatch);
        return;
    }
    const uint i = First + lane;
    const float3 look = NormalizedGuide(Target - Eye);
    const float3 side = NormalizedGuide(cross(look, Up)), vertical = cross(side, look);
    const float halfY = tan(FieldOfView * Pi / (2 * DegreesPerHalfTurn));
    const float halfX = halfY * float(ImageWidth) / float(ImageHeight);
    // Center sampling is intentionally unjittered and uses the full film.
    static const float PixelCenter = 0.5f;
    const float u = (float(i % TileWidth + Left) + PixelCenter) / float(ImageWidth);
    const float v = (float(i / TileWidth + Top) + PixelCenter) / float(ImageHeight);
    const float3 direction =
        NormalizedGuide(look + (2 * u - 1) * halfX * side + (1 - 2 * v) * halfY * vertical);
    if (any(!isfinite(direction)) || !(dot(direction, direction) > 0))
    {
        InterlockedOr(Failure[0], InvalidValue);
        return;
    }
    EncodedGuideRay ray;
    ray.EyeMinimum = float4(Eye, NearDistance);
    ray.DirectionMaximum = float4(direction, FarDistance);
    CameraRays[i] = ray;
}
[shader("raygeneration")] void ResolveGeometryGuideHits()
{
    const uint lane = DispatchRaysIndex().x;
    if (lane >= Population)
        return;
    if (!GuideDispatchValid())
    {
        InterlockedOr(Failure[0], InvalidDispatch);
        return;
    }
    const uint i = First + lane;
    const EncodedGuideRay ray = Rays[i];
    const NativeGuideHit hit = Intersections[i];
    GeometryGuide value = (GeometryGuide)0;
    value.Direction = ray.DirectionMaximum.xyz;
    value.Primitive = NoPrimitive;
    value.Flags = SkyFlag;
    if (hit.Flags & SurfaceFlag)
    {
        const double t = double(hit.Distance);
        value.Point = float3(double(ray.EyeMinimum.x) + double(value.Direction.x) * t,
                             double(ray.EyeMinimum.y) + double(value.Direction.y) * t,
                             double(ray.EyeMinimum.z) + double(value.Direction.z) * t);
        value.Normal = NormalizedGuide(hit.WindingNormal);
        value.CameraDepth = hit.Distance * dot(value.Direction, NormalizedGuide(Target - Eye));
        value.Distance = hit.Distance;
        value.Primitive = hit.Primitive;
        value.Material = hit.Material;
        value.Flags = SurfaceFlag | (hit.Flags & FrontFlag);
        if (any(!isfinite(value.Point)) || any(!isfinite(value.Normal)) ||
            !(dot(value.Normal, value.Normal) > 0) || !isfinite(value.CameraDepth) ||
            !(value.CameraDepth > 0) || !isfinite(hit.Distance) || hit.Distance <= ray.EyeMinimum.w ||
            hit.Distance >= ray.DirectionMaximum.w || hit.Primitive == NoPrimitive)
        {
            InterlockedOr(Failure[0], InvalidValue);
            return;
        }
    }
    Result[i] = value;
}
