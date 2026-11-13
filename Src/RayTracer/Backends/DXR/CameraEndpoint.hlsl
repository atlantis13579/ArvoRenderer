// Independent DXR camera response and connections. PBRT 3e, section 16.1.1.
// Uniform full-film area implies p(w)=1/(A*cos^3), We=p(lens)*p(w)/cos.
// This matches the demo camera's normalized exposure, not photographic falloff.
// Raster coordinates cover the entire image; per-pixel/splat scaling is external.
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
    CameraEndpointEvaluation r;
    r.Raster = float2(0, 0);
    r.Importance = 0;
    r.PositionPdf = 0;
    r.DirectionPdf = 0;
    r.Flags = 0;
    return r;
}
CameraEndpointSample EmptyCameraEndpoint()
{
    CameraEndpointSample r;
    r.Origin = float3(0, 0, 0);
    r.Direction = float3(0, 0, 0);
    r.LensPoint = float2(0, 0);
    r.Raster = float2(0, 0);
    r.Importance = 0;
    r.PositionPdf = 0;
    r.DirectionPdf = 0;
    r.Weight = 0;
    r.Flags = 0;
    return r;
}
CameraConnectionSample EmptyCameraConnection()
{
    CameraConnectionSample r;
    r.Position = float3(0, 0, 0);
    r.Direction = float3(0, 0, 0);
    r.LensPoint = float2(0, 0);
    r.Raster = float2(0, 0);
    r.Distance = 0;
    r.Weight = 0;
    r.PositionPdf = 0;
    r.DirectionPdf = 0;
    r.Flags = 0;
    return r;
}
bool CameraInRaster(float2 uv)
{
    return uv.x >= 0 && uv.x < 1 && uv.y >= 0 && uv.y < 1;
}
float2 CameraPixelRaster(uint x, uint y, uint width, uint height, float2 offset)
{
    if (x >= width || y >= height || !CameraInRaster(offset)) return float2(-1, -1);
    // Float additions near a large image border may equal the image extent.
    // Repair only coordinates generated from already validated pixel samples.
    static const float FractionSteps = 16777216.0f;
    static const float LastRasterValue = 1 - 1 / FractionSteps;
    float horizontal = (float(x) + offset.x) / float(width);
    float vertical = (float(y) + offset.y) / float(height);
    return float2(horizontal < LastRasterValue ? horizontal : LastRasterValue,
                  vertical < LastRasterValue ? vertical : LastRasterValue);
}
float CameraAperturePdf(CameraEndpointFrame frame)
{
    if (!(frame.HalfWidth > 0 && frame.HalfHeight > 0 && frame.LensRadius >= 0))
        return 0;
    if (frame.LensRadius == 0)
        return 1;
    if (!(frame.FocusDistance > 0))
        return 0;
    float p = 1 / (CameraEndpointPi * frame.LensRadius * frame.LensRadius);
    if (!(p > 0) || !isfinite(p))
        return 0;
    return p;
}
float CameraLength(float3 v)
{
    float largest = max(abs(v.x), max(abs(v.y), abs(v.z)));
    if (!(largest > 0) || !isfinite(largest))
        return 0;
    float3 scaled = v / largest;
    return sqrt(dot(scaled, scaled)) * largest;
}
float2 CameraAperturePoint(CameraEndpointFrame frame, float2 variates)
{
    float a = variates.x * 2 - 1, b = variates.y * 2 - 1;
    if (frame.LensRadius == 0 || (a == 0 && b == 0))
        return float2(0, 0);
    bool majorX = abs(a) > abs(b);
    float radius = majorX ? a : b;
    float angle =
        majorX ? (CameraEndpointPi / 4) * b / a : CameraEndpointPi / 2 - (CameraEndpointPi / 4) * a / b;
    return float2(radius * cos(angle), radius * sin(angle)) * frame.LensRadius;
}
CameraEndpointEvaluation CameraImportance(CameraEndpointFrame frame, float2 raster, float cosine)
{
    CameraEndpointEvaluation evaluation = EmptyCameraEvaluation();
    float position = CameraAperturePdf(frame);
    if (!(position > 0 && cosine > 0))
        return evaluation;
    float planeArea = 4 * frame.HalfWidth * frame.HalfHeight;
    float direction = 1 / (planeArea * cosine * cosine * cosine), value = position * direction / cosine;
    if (!(direction > 0 && value > 0) || !isfinite(direction) || !isfinite(value))
        return evaluation;
    evaluation.Raster = raster;
    evaluation.Importance = value;
    evaluation.DirectionPdf = direction;
    evaluation.PositionPdf = frame.LensRadius == 0 ? 0 : position;
    evaluation.Flags = CameraEndpointValid | (frame.LensRadius == 0 ? CameraDeltaPosition : 0);
    return evaluation;
}
// Orthonormal frame, known point on lens support, normalized outgoing direction.
// Pinhole position density is zero here; the sampled discrete mass is one.
CameraEndpointEvaluation EvaluateCameraEndpoint(CameraEndpointFrame frame, float2 lens, float3 outgoing)
{
    float cosTheta = dot(frame.Forward, outgoing);
    if (!(cosTheta > 0) || !(CameraAperturePdf(frame) > 0))
        return EmptyCameraEvaluation();
    float distance = frame.LensRadius == 0 ? 1 : frame.FocusDistance;
    float projectedX = lens.x / distance + dot(frame.Right, outgoing) / cosTheta;
    float projectedY = lens.y / distance + dot(frame.Up, outgoing) / cosTheta;
    float2 uv = float2((1 + projectedX / frame.HalfWidth) * 0.5f, (1 - projectedY / frame.HalfHeight) * 0.5f);
    if (!CameraInRaster(uv))
        return EmptyCameraEvaluation();
    return CameraImportance(frame, uv, cosTheta);
}
CameraEndpointSample SampleCameraEndpoint(CameraEndpointFrame frame, float2 raster, float2 apertureVariates)
{
    CameraEndpointSample sample = EmptyCameraEndpoint();
    if (!CameraInRaster(raster) || !CameraInRaster(apertureVariates) || !(CameraAperturePdf(frame) > 0))
        return sample;
    float2 aperture = CameraAperturePoint(frame, apertureVariates);
    float3 displacement = frame.Right * aperture.x + frame.Up * aperture.y;
    float3 direction = frame.Forward + frame.Right * ((raster.x * 2 - 1) * frame.HalfWidth) +
                       frame.Up * ((1 - raster.y * 2) * frame.HalfHeight);
    if (frame.LensRadius > 0)
        direction = direction * frame.FocusDistance - displacement;
    float magnitude = CameraLength(direction);
    if (!(magnitude > 0) || !isfinite(magnitude))
        return sample;
    direction = direction / magnitude;
    CameraEndpointEvaluation importance = CameraImportance(frame, raster, dot(frame.Forward, direction));
    if (!importance.Flags)
        return sample;
    sample.Origin = frame.Position + displacement;
    sample.Direction = direction;
    sample.LensPoint = aperture;
    sample.Raster = raster;
    sample.Importance = importance.Importance;
    sample.PositionPdf = frame.LensRadius == 0 ? 1 : importance.PositionPdf;
    sample.DirectionPdf = importance.DirectionPdf;
    sample.Weight = 1;
    sample.Flags = importance.Flags;
    return sample;
}
CameraConnectionSample SampleCameraConnection(CameraEndpointFrame frame, float3 scenePoint, float2 variates)
{
    CameraConnectionSample connection = EmptyCameraConnection();
    if (!CameraInRaster(variates) || !(CameraAperturePdf(frame) > 0))
        return connection;
    float2 lens = CameraAperturePoint(frame, variates);
    float3 offset = frame.Right * lens.x + frame.Up * lens.y;
    float3 outward = (scenePoint - frame.Position) - offset;
    float distance = CameraLength(outward);
    if (!(distance > 0) || !isfinite(distance))
        return connection;
    outward = outward / distance;
    CameraEndpointEvaluation response = EvaluateCameraEndpoint(frame, lens, outward);
    if (!response.Flags)
        return connection;
    bool delta = frame.LensRadius == 0;
    float cosine = dot(frame.Forward, outward);
    float pdf = delta ? 1 : (response.PositionPdf * distance) * (distance / cosine);
    float weight = (response.DirectionPdf / distance) / distance;
    if (!(pdf > 0 && weight > 0) || !isfinite(pdf) || !isfinite(weight))
        return connection;
    connection.Position = frame.Position + offset;
    connection.Direction = -outward;
    connection.LensPoint = lens;
    connection.Raster = response.Raster;
    connection.Distance = distance;
    connection.Weight = weight;
    connection.PositionPdf = delta ? 1 : response.PositionPdf;
    connection.DirectionPdf = pdf;
    connection.Flags = response.Flags | (delta ? CameraDeltaDirection : 0);
    // Delta angular mass is 1. Weight already includes lens cosine / distance^2;
    // scene visibility, BSDF and scene cosine must be supplied by the integrator.
    return connection;
}
