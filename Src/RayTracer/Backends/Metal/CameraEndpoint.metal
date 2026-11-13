// Independent Metal perspective endpoints. PBRT 3e, section 16.1.1,
// "Sampling Cameras". Full-film p(w)=1/(A*cos^3), A=4*HalfWidth*HalfHeight.
// Normalized importance We=p(lens)*p(w)/cos cancels aperture/exposure changes.
// Pixel-conditioned direction PDFs and light-tracing splat scaling are external.
constant float CameraEndpointPi = 3.14159265358979323846f;
constant uint CameraEndpointValid = 1, CameraDeltaPosition = 2, CameraDeltaDirection = 4;
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
bool CameraValidRaster(float2 uv)
{
    return uv.x >= 0 && uv.x < 1 && uv.y >= 0 && uv.y < 1;
}
float2 CameraPixelRaster(uint column, uint row, uint columns, uint rows, float2 subpixel)
{
    if (column >= columns || row >= rows || !CameraValidRaster(subpixel)) return float2(-1);
    const float FloatFractionRange = 16777216.0f;
    const float UpperRasterBound = 1 - 1 / FloatFractionRange;
    float u = (float(column) + subpixel.x) / float(columns);
    float v = (float(row) + subpixel.y) / float(rows);
    // MSL uses float here. Clamp the rounded upper edge without changing the
    // endpoint API's rejection of caller-supplied coordinates outside [0,1).
    return float2(min(u, UpperRasterBound), min(v, UpperRasterBound));
}
float CameraLensProbability(CameraEndpointFrame camera)
{
    if (!(camera.HalfWidth > 0 && camera.HalfHeight > 0 && camera.LensRadius >= 0))
        return 0;
    if (camera.LensRadius == 0)
        return 1;
    if (!(camera.FocusDistance > 0))
        return 0;
    float probability = 1 / (CameraEndpointPi * camera.LensRadius * camera.LensRadius);
    return probability > 0 && isfinite(probability) ? probability : 0;
}
float CameraLength(float3 v)
{
    float largest = max(fabs(v.x), max(fabs(v.y), fabs(v.z)));
    if (!(largest > 0) || !isfinite(largest))
        return 0;
    float3 scaled = v / largest;
    return largest * sqrt(dot(scaled, scaled));
}
float2 CameraLensPosition(CameraEndpointFrame camera, float2 u)
{
    float a = 2 * u.x - 1, b = 2 * u.y - 1;
    if (camera.LensRadius == 0 || (a == 0 && b == 0))
        return float2(0);
    bool xMajor = fabs(a) > fabs(b);
    float radius = xMajor ? a : b;
    float phi =
        xMajor ? (CameraEndpointPi / 4) * (b / a) : CameraEndpointPi / 2 - (CameraEndpointPi / 4) * (a / b);
    return (camera.LensRadius * radius) * float2(cos(phi), sin(phi));
}
CameraEndpointEvaluation CameraEvaluateResponse(CameraEndpointFrame camera, float2 raster, float cosine)
{
    CameraEndpointEvaluation response{};
    float position = CameraLensProbability(camera);
    if (!(position > 0 && cosine > 0))
        return response;
    float area = 4 * camera.HalfWidth * camera.HalfHeight;
    float direction = 1 / (area * cosine * cosine * cosine), importance = position * direction / cosine;
    if (!(direction > 0 && importance > 0) || !isfinite(direction) || !isfinite(importance))
        return response;
    response.Raster = raster;
    response.Importance = importance;
    response.DirectionPdf = direction;
    response.PositionPdf = camera.LensRadius == 0 ? 0 : position;
    response.Flags = CameraEndpointValid | (camera.LensRadius == 0 ? CameraDeltaPosition : 0);
    return response;
}
// Known lens support, orthonormal frame and unit camera-to-scene direction.
// Lens coordinates are world units along Right/Up; pinhole coordinates are zero.
CameraEndpointEvaluation EvaluateCameraEndpoint(CameraEndpointFrame camera, float2 lens, float3 direction)
{
    if (!(CameraLensProbability(camera) > 0))
        return {};
    float cosine = dot(camera.Forward, direction);
    if (!(cosine > 0))
        return {};
    float focus = camera.LensRadius == 0 ? 1 : camera.FocusDistance;
    float projectedX = lens.x / focus + dot(camera.Right, direction) / cosine;
    float projectedY = lens.y / focus + dot(camera.Up, direction) / cosine;
    float2 raster =
        float2(0.5f + projectedX / (2 * camera.HalfWidth), 0.5f - projectedY / (2 * camera.HalfHeight));
    if (!CameraValidRaster(raster))
        return {};
    return CameraEvaluateResponse(camera, raster, cosine);
}
CameraEndpointSample SampleCameraEndpoint(CameraEndpointFrame camera, float2 raster, float2 lensU)
{
    if (!CameraValidRaster(raster) || !CameraValidRaster(lensU) || !(CameraLensProbability(camera) > 0))
        return {};
    float2 lens = CameraLensPosition(camera, lensU);
    float3 shift = camera.Right * lens.x + camera.Up * lens.y;
    float3 pinhole = camera.Forward + camera.Right * ((2 * raster.x - 1) * camera.HalfWidth) +
                     camera.Up * ((1 - 2 * raster.y) * camera.HalfHeight);
    float3 direction = camera.LensRadius == 0 ? pinhole : pinhole * camera.FocusDistance - shift;
    float magnitude = CameraLength(direction);
    if (!(magnitude > 0) || !isfinite(magnitude))
        return {};
    direction /= magnitude;
    CameraEndpointEvaluation response =
        CameraEvaluateResponse(camera, raster, dot(direction, camera.Forward));
    if (response.Flags == 0)
        return {};
    return {camera.Position + shift,
            direction,
            lens,
            raster,
            response.Importance,
            camera.LensRadius == 0 ? 1 : response.PositionPdf,
            response.DirectionPdf,
            1,
            response.Flags};
}
// A rejected connection consumes its lens draw. Visibility, scene BSDF and
// scene cosine belong to the caller. Pinhole angular samples have unit mass;
// the lens cosine/distance^2 factor remains in Weight, not in a fake density.
CameraConnectionSample SampleCameraConnection(CameraEndpointFrame camera, float3 scenePoint, float2 lensU)
{
    if (!CameraValidRaster(lensU) || !(CameraLensProbability(camera) > 0))
        return {};
    float2 lens = CameraLensPosition(camera, lensU);
    float3 shift = camera.Right * lens.x + camera.Up * lens.y;
    float3 outward = (scenePoint - camera.Position) - shift;
    float distance = CameraLength(outward);
    if (!(distance > 0) || !isfinite(distance))
        return {};
    outward /= distance;
    CameraEndpointEvaluation response = EvaluateCameraEndpoint(camera, lens, outward);
    if (response.Flags == 0)
        return {};
    bool isPinhole = camera.LensRadius == 0;
    float cosine = dot(camera.Forward, outward);
    float pdf = isPinhole ? 1 : (response.PositionPdf * distance) * (distance / cosine);
    float weight = response.DirectionPdf / distance / distance;
    if (!(pdf > 0 && weight > 0) || !isfinite(pdf) || !isfinite(weight))
        return {};
    return {camera.Position + shift,
            -outward,
            lens,
            response.Raster,
            distance,
            weight,
            isPinhole ? 1 : response.PositionPdf,
            pdf,
            response.Flags | (isPinhole ? CameraDeltaDirection : 0)};
}
