// Thin-lens aperture averaging, independently implemented for Compute.
// PBRT v4, "Projective Camera Models":
// https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models
// FocusDistance is axial scene distance. This normalized camera does not model
// f-number exposure or vignetting. The square-to-disk map preserves uniform area.
static const uint LensSampleDimension = 0x4c454e53u; // "LENS", not a transport dimension.
void ApplyThinLens(inout RayData ray, float2 sample)
{
    if (LensRadius == 0) return;
    float2 square = 2 * sample - 1;
    float2 disk = float2(0, 0);
    if (any(square != 0))
    {
        float radius, azimuth;
        if (abs(square.x) > abs(square.y))
        {
            radius = square.x;
            azimuth = (Pi / 4) * square.y / square.x;
        }
        else
        {
            radius = square.y;
            azimuth = Pi / 2 - (Pi / 4) * square.x / square.y;
        }
        disk = radius * float2(cos(azimuth), sin(azimuth));
    }
    float3 apertureOffset = LensRadius * (disk.x * CameraRight + disk.y * CameraUp);
    float axialCosine = dot(ray.Direction, CameraForward);
    float3 focusOffset = ray.Direction * (FocusDistance / axialCosine);
    ray.Origin = CameraPosition + apertureOffset;
    ray.Direction = normalize(focusOffset - apertureOffset);
}
