// Native Metal thin-lens camera, with an area-preserving concentric aperture map.
// PBRT v4, "Projective Camera Models":
// https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models
// The aperture average keeps exposure fixed. FocusDistance is scene distance
// along the view axis, and radius zero is an exact pinhole limit.
constant uint LensSampleDimension = 0x4c454e53u; // "LENS", independent of raster/transport.
void ApplyThinLens(thread RayData& ray, constant FrameConstants& frame, float2 sample)
{
    if (frame.LensRadius == 0) return;
    const float2 square = sample * 2 - 1;
    float2 disk = float2(0);
    if (square.x != 0 || square.y != 0)
    {
        float signedRadius, polarAngle;
        if (abs(square.x) > abs(square.y))
        {
            signedRadius = square.x;
            polarAngle = (M_PI_F / 4) * (square.y / square.x);
        }
        else
        {
            signedRadius = square.y;
            polarAngle = M_PI_F / 2 - (M_PI_F / 4) * (square.x / square.y);
        }
        disk = float2(cos(polarAngle), sin(polarAngle)) * signedRadius;
    }
    const float3 lensOffset = frame.LensRadius *
        (float3(frame.CameraRight) * disk.x + float3(frame.CameraUp) * disk.y);
    const float distance = frame.FocusDistance / dot(ray.Direction, float3(frame.CameraForward));
    const float3 focusOffset = distance * ray.Direction;
    ray.Origin = float3(frame.CameraPosition) + lensOffset;
    ray.Direction = normalize(focusOffset - lensOffset);
}
