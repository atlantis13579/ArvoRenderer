// Independent Compute camera endpoints. PBRT 3e, section 16.1.1, "Sampling Cameras".
// p(direction)=1/(A*cos^3), A=4*HalfWidth*HalfHeight at unit image depth.
// We=p(lens)*p(direction)/cos gives the existing aperture-averaged exposure.
// Raster covers the full image [0,1)^2. Pixel/tile normalization is external.
static const float CameraEndpointPi = 3.14159265358979323846f;
static const uint CameraEndpointValid = 1, CameraDeltaPosition = 2, CameraDeltaDirection = 4;
struct CameraEndpointFrame
{
    float3 Position, Forward, Right, Up;
    float HalfWidth, HalfHeight, LensRadius, FocusDistance;
};
struct CameraEndpointEvaluation
{
    float2 Raster;
    float Importance, PositionPdf, DirectionPdf;
    uint Flags;
};
struct CameraEndpointSample
{
    float3 Origin, Direction;
    float2 LensPoint, Raster;
    float Importance, PositionPdf, DirectionPdf, Weight;
    uint Flags;
};
struct CameraConnectionSample
{
    float3 Position, Direction;
    float2 LensPoint, Raster;
    float Distance, Weight, PositionPdf, DirectionPdf;
    uint Flags;
};
CameraEndpointEvaluation EmptyCameraEvaluation()
{
    CameraEndpointEvaluation result;
    result.Raster = float2(0, 0);
    result.Importance = 0;
    result.PositionPdf = 0;
    result.DirectionPdf = 0;
    result.Flags = 0;
    return result;
}
CameraEndpointSample EmptyCameraEndpoint()
{
    CameraEndpointSample result;
    result.Origin = result.Direction = float3(0, 0, 0);
    result.LensPoint = result.Raster = float2(0, 0);
    result.Importance = result.PositionPdf = result.DirectionPdf = result.Weight = 0;
    result.Flags = 0;
    return result;
}
CameraConnectionSample EmptyCameraConnection()
{
    CameraConnectionSample result;
    result.Position = result.Direction = float3(0, 0, 0);
    result.LensPoint = result.Raster = float2(0, 0);
    result.Distance = result.Weight = result.PositionPdf = result.DirectionPdf = 0;
    result.Flags = 0;
    return result;
}
bool CameraUnitSquare(float2 u)
{
    return u.x >= 0 && u.x < 1 && u.y >= 0 && u.y < 1;
}
float2 CameraPixelRaster(uint x, uint y, uint width, uint height, float2 jitter)
{
    if (x >= width || y >= height || !CameraUnitSquare(jitter)) return float2(-1, -1);
    static const float MantissaRange = 16777216.0f;
    static const float RasterUpper = 1 - 1 / MantissaRange;
    // Form pixel positions in double, then repair a float rounding onto the
    // excluded upper edge. Actual out-of-support requests remain invalid.
    float u = float((double(x) + double(jitter.x)) / double(width));
    float v = float((double(y) + double(jitter.y)) / double(height));
    return float2(min(u, RasterUpper), min(v, RasterUpper));
}
float CameraLensDensity(CameraEndpointFrame camera)
{
    if (!(camera.HalfWidth > 0 && camera.HalfHeight > 0 && camera.LensRadius >= 0))
        return 0;
    if (camera.LensRadius == 0)
        return 1;
    if (!(camera.FocusDistance > 0))
        return 0;
    float density = 1 / (CameraEndpointPi * camera.LensRadius * camera.LensRadius);
    return density > 0 && isfinite(density) ? density : 0;
}
float CameraVectorLength(float3 v)
{
    float scale = max(abs(v.x), max(abs(v.y), abs(v.z)));
    if (!(scale > 0) || !isfinite(scale))
        return 0;
    float3 unitScale = v / scale;
    return scale * sqrt(dot(unitScale, unitScale));
}
float2 CameraLensPoint(CameraEndpointFrame camera, float2 u)
{
    float x = 2 * u.x - 1, y = 2 * u.y - 1;
    if (camera.LensRadius == 0 || (x == 0 && y == 0))
        return float2(0, 0);
    float radius, angle;
    if (abs(x) > abs(y))
    {
        radius = x;
        angle = (CameraEndpointPi / 4) * (y / x);
    }
    else
    {
        radius = y;
        angle = CameraEndpointPi / 2 - (CameraEndpointPi / 4) * (x / y);
    }
    return float2(cos(angle), sin(angle)) * (camera.LensRadius * radius);
}
CameraEndpointEvaluation CameraResponse(CameraEndpointFrame camera, float2 raster, float cosine)
{
    CameraEndpointEvaluation result = EmptyCameraEvaluation();
    float lensPdf = CameraLensDensity(camera);
    if (!(lensPdf > 0 && cosine > 0))
        return result;
    float area = 4 * camera.HalfWidth * camera.HalfHeight;
    float rayPdf = 1 / (area * cosine * cosine * cosine), importance = lensPdf * rayPdf / cosine;
    if (!(rayPdf > 0 && importance > 0) || !isfinite(rayPdf) || !isfinite(importance))
        return result;
    result.Raster = raster;
    result.Importance = importance;
    result.DirectionPdf = rayPdf;
    result.PositionPdf = camera.LensRadius == 0 ? 0 : lensPdf;
    result.Flags = CameraEndpointValid | (camera.LensRadius == 0 ? CameraDeltaPosition : 0);
    return result;
}
// Lens point is already on aperture support, in Right/Up world units. Frame
// axes and ray direction are unit. Queries contain continuous densities only.
CameraEndpointEvaluation EvaluateCameraEndpoint(CameraEndpointFrame camera, float2 lens, float3 direction)
{
    float cosine = dot(direction, camera.Forward);
    if (!(cosine > 0) || !(CameraLensDensity(camera) > 0))
        return EmptyCameraEvaluation();
    float focus = camera.LensRadius == 0 ? 1 : camera.FocusDistance;
    float x = lens.x / focus + dot(direction, camera.Right) / cosine;
    float y = lens.y / focus + dot(direction, camera.Up) / cosine;
    float2 raster = float2(0.5f + x / (2 * camera.HalfWidth), 0.5f - y / (2 * camera.HalfHeight));
    if (!CameraUnitSquare(raster))
        return EmptyCameraEvaluation();
    return CameraResponse(camera, raster, cosine);
}
CameraEndpointSample SampleCameraEndpoint(CameraEndpointFrame camera, float2 raster, float2 lensVariates)
{
    CameraEndpointSample result = EmptyCameraEndpoint();
    if (!CameraUnitSquare(raster) || !CameraUnitSquare(lensVariates) || !(CameraLensDensity(camera) > 0))
        return result;
    float2 lens = CameraLensPoint(camera, lensVariates);
    float3 offset = camera.Right * lens.x + camera.Up * lens.y;
    float3 axis = camera.Forward + camera.Right * ((2 * raster.x - 1) * camera.HalfWidth) +
                  camera.Up * ((1 - 2 * raster.y) * camera.HalfHeight);
    float3 ray = camera.LensRadius == 0 ? axis : camera.FocusDistance * axis - offset;
    float distance = CameraVectorLength(ray);
    if (!(distance > 0) || !isfinite(distance))
        return result;
    float3 direction = ray / distance;
    CameraEndpointEvaluation response = CameraResponse(camera, raster, dot(direction, camera.Forward));
    if (response.Flags == 0)
        return result;
    result.Origin = camera.Position + offset;
    result.Direction = direction;
    result.LensPoint = lens;
    result.Raster = raster;
    result.Importance = response.Importance;
    result.PositionPdf = camera.LensRadius == 0 ? 1 : response.PositionPdf;
    result.DirectionPdf = response.DirectionPdf;
    result.Weight = 1;
    result.Flags = response.Flags;
    return result;
}
// Direction points from scene reference to lens. Weight includes lens geometry;
// visibility and the scene BSDF/cosine are external. Never retry a null draw.
CameraConnectionSample SampleCameraConnection(CameraEndpointFrame camera, float3 reference, float2 variates)
{
    CameraConnectionSample result = EmptyCameraConnection();
    if (!CameraUnitSquare(variates) || !(CameraLensDensity(camera) > 0))
        return result;
    float2 lens = CameraLensPoint(camera, variates);
    float3 offset = camera.Right * lens.x + camera.Up * lens.y,
           segment = (reference - camera.Position) - offset;
    float distance = CameraVectorLength(segment);
    if (!(distance > 0) || !isfinite(distance))
        return result;
    float3 outward = segment / distance;
    CameraEndpointEvaluation response = EvaluateCameraEndpoint(camera, lens, outward);
    if (response.Flags == 0)
        return result;
    bool pinhole = camera.LensRadius == 0;
    float cosine = dot(camera.Forward, outward);
    float density = pinhole ? 1 : (response.PositionPdf * distance) * (distance / cosine);
    float weight = response.DirectionPdf / distance / distance;
    if (!(density > 0 && weight > 0) || !isfinite(density) || !isfinite(weight))
        return result;
    result.Position = camera.Position + offset;
    result.Direction = -outward;
    result.LensPoint = lens;
    result.Raster = response.Raster;
    result.Distance = distance;
    result.Weight = weight;
    result.PositionPdf = pinhole ? 1 : response.PositionPdf;
    result.DirectionPdf = density;
    result.Flags = response.Flags | (pinhole ? CameraDeltaDirection : 0);
    return result;
}
