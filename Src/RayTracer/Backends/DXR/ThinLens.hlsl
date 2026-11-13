// DXR owns this camera estimator; no software/backend camera implementation is
// called. Concentric disk sampling avoids rejection and preserves QMC samples.
// PBRT v4, "Projective Camera Models":
// https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models
// Radius zero leaves the exact pinhole ray intact. Finite radius averages lens
// radiance with normalized exposure; FocusDistance names an axial focus plane.
static const uint LensSampleDimension = 0x4c454e53u; // ASCII "LENS".
void ApplyThinLens(inout float3 origin, inout float3 direction, float2 lensSample)
{
    if (LensRadius == 0) return;
    float a = lensSample.x * 2 - 1, b = lensSample.y * 2 - 1;
    float2 aperture = 0;
    if (a != 0 || b != 0)
    {
        bool majorX = abs(a) > abs(b);
        float radius = majorX ? a : b;
        float theta;
        if (majorX) theta = (Pi / 4) * b / a;
        else theta = Pi / 2 - (Pi / 4) * a / b;
        aperture = float2(radius * cos(theta), radius * sin(theta));
    }
    float3 offset = CameraRight * (LensRadius * aperture.x) + CameraUp * (LensRadius * aperture.y);
    float3 towardFocus = direction * (FocusDistance / dot(CameraForward, direction));
    origin = CameraPosition + offset;
    direction = normalize(towardFocus - offset);
}
