// Independent CUDA thin lens. Uniform aperture-area samples converge to the
// aperture average of radiance, retaining exposure when the radius changes.
// PBRT v4, "Projective Camera Models" (focus plane, not focal length):
// https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models
constexpr unsigned int LensSampleDimension = 0x4c454e53u; // "LENS" sample namespace.
__device__ void ApplyThinLens(Ray& ray, const FrameConstants& frame, float2 variates)
{
    if (frame.LensRadius == 0) return;
    const float u = 2 * variates.x - 1, v = 2 * variates.y - 1;
    float apertureX = 0, apertureY = 0;
    if (u != 0 || v != 0)
    {
        float radial, angle;
        if (fabsf(u) > fabsf(v))
        {
            radial = u;
            angle = (CUDART_PI_F / 4) * (v / u);
        }
        else
        {
            radial = v;
            angle = CUDART_PI_F / 2 - (CUDART_PI_F / 4) * (u / v);
        }
        apertureX = radial * cosf(angle);
        apertureY = radial * sinf(angle);
    }
    const Vec3 displacement = Multiply(Add(Multiply(frame.CameraRight, apertureX),
                                          Multiply(frame.CameraUp, apertureY)), frame.LensRadius);
    const float planeParameter = frame.FocusDistance / Dot(ray.Direction, frame.CameraForward);
    const Vec3 focusVector = Multiply(ray.Direction, planeParameter);
    ray.Origin = Add(frame.CameraPosition, displacement);
    ray.Direction = Normalize(Subtract(focusVector, displacement));
}
