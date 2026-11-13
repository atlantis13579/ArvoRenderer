// Independent native CUDA homogeneous medium. PBRT v3, sections 11.3/15.2.
// RGB exponential proposals include the endpoint mass. The selected channel
// chooses a distance, but the complete RGB mixture determines throughput.
#include <cuda_runtime.h>
#include <cmath>
struct alignas(16) MediumParameters
{
    float3 Minimum;
    unsigned Enabled;
    float3 Maximum;
    float Anisotropy;
    float3 Absorption;
    unsigned Padding0;
    float3 Scattering;
    unsigned Padding1;
};
struct MediumFlight
{
    float3 Weight;
    float Distance;
    unsigned Scattered;
};
static_assert(sizeof(MediumParameters) == 64);
constexpr float MediumMaximumUniform = 0.9999999403953552f;
constexpr double MediumFourPi = 4 * 3.14159265358979323846;
__device__ float MediumDot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 MediumCross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ float3 MediumUnit(float3 a)
{
    const float inverse = 1 / sqrtf(MediumDot(a, a));
    return make_float3(a.x * inverse, a.y * inverse, a.z * inverse);
}
__device__ bool MediumClip(MediumParameters region, float3 origin, float3 direction, float minimum,
                           float maximum, float& first, float& last)
{
    first = fmaxf(minimum, 0);
    last = maximum;
    if (!region.Enabled)
        return false;
    const float p[]{origin.x, origin.y, origin.z}, d[]{direction.x, direction.y, direction.z};
    const float low[]{region.Minimum.x, region.Minimum.y, region.Minimum.z};
    const float high[]{region.Maximum.x, region.Maximum.y, region.Maximum.z};
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        if (d[axis] == 0)
        {
            if (p[axis] < low[axis] || p[axis] > high[axis])
                return false;
        }
        else
        {
            const float nearPlane = (low[axis] - p[axis]) / d[axis],
                        farPlane = (high[axis] - p[axis]) / d[axis];
            first = fmaxf(first, fminf(nearPlane, farPlane));
            last = fminf(last, fmaxf(nearPlane, farPlane));
        }
    }
    return last > first;
}
__device__ float3 MediumBeer(float3 extinction, float length)
{
    return make_float3(float(exp(-double(extinction.x) * length)), float(exp(-double(extinction.y) * length)),
                       float(exp(-double(extinction.z) * length)));
}
__device__ float3 MediumTransmission(MediumParameters region, float3 origin, float3 direction, float minimum,
                                     float maximum)
{
    float start, end;
    if (!MediumClip(region, origin, direction, minimum, maximum, start, end))
        return make_float3(1, 1, 1);
    const float3 extinction =
        make_float3(region.Absorption.x + region.Scattering.x, region.Absorption.y + region.Scattering.y,
                    region.Absorption.z + region.Scattering.z);
    return MediumBeer(extinction, (end - start) * sqrtf(MediumDot(direction, direction)));
}
__device__ MediumFlight MediumFreeFlight(MediumParameters region, float3 origin, float3 direction,
                                         float minimum, float maximum, float2 random)
{
    MediumFlight result{make_float3(1, 1, 1), 0, 0};
    float first, last;
    if (!MediumClip(region, origin, direction, minimum, maximum, first, last))
        return result;
    const float speed = sqrtf(MediumDot(direction, direction));
    if (!(speed > 0))
        return result;
    const float limit = (last - first) * speed;
    const float3 extinction =
        make_float3(region.Absorption.x + region.Scattering.x, region.Absorption.y + region.Scattering.y,
                    region.Absorption.z + region.Scattering.z);
    if (region.Scattering.x == 0 && region.Scattering.y == 0 && region.Scattering.z == 0)
    {
        result.Weight = MediumBeer(extinction, limit);
        return result;
    }
    constexpr unsigned Channels = 3;
    const unsigned channel = unsigned(fminf(MediumMaximumUniform, fmaxf(0, random.x)) * Channels);
    const float sigma[]{extinction.x, extinction.y, extinction.z};
    const double uniform = fminf(MediumMaximumUniform, fmaxf(0, random.y));
    const double freePath = sigma[channel] > 0 ? -log1p(-uniform) / sigma[channel] : double(limit);
    result.Scattered = freePath < limit;
    const float length = result.Scattered ? float(freePath) : limit;
    result.Distance = first + length / speed;
    const float3 tr = MediumBeer(extinction, length);
    const double pdf =
        result.Scattered
            ? (double(tr.x) * sigma[0] + double(tr.y) * sigma[1] + double(tr.z) * sigma[2]) / Channels
            : (double(tr.x) + tr.y + tr.z) / Channels;
    const float3 factor = result.Scattered ? region.Scattering : make_float3(1, 1, 1);
    result.Weight =
        pdf > 0 ? make_float3(float(double(tr.x) * factor.x / pdf), float(double(tr.y) * factor.y / pdf),
                              float(double(tr.z) * factor.z / pdf))
                : make_float3(0, 0, 0);
    return result;
}
__device__ float MediumPhase(float cosine, float anisotropy)
{
    const double g = anisotropy, c = fmin(1.0, fmax(-1.0, double(cosine)));
    const double denominator = 1 + g * g - 2 * g * c;
    return float((1 - g) * (1 + g) / (MediumFourPi * denominator * sqrt(denominator)));
}
__device__ float3 MediumSampleDirection(float3 incoming, float anisotropy, float2 random, bool importance,
                                        float& pdf)
{
    const double g = importance ? anisotropy : 0;
    const double v = 2 * double(fminf(MediumMaximumUniform, fmaxf(0, random.x))) - 1;
    const double d = 1 + g * v;
    // Polynomial HG inversion remains defined at g=0; no phase approximation.
    const double c =
        fmin(1.0, fmax(-1.0, (2 * v * (1 + g * g) + g * (3 - g * g + v * v * (1 + g * g))) / (2 * d * d)));
    const double phi = MediumFourPi / 2 * fminf(MediumMaximumUniform, fmaxf(0, random.y));
    const double sine = sqrt(fmax(0.0, 1 - c * c));
    incoming = MediumUnit(incoming);
    constexpr float ParallelThreshold = 0.999f;
    const float3 helper = fabsf(incoming.y) < ParallelThreshold ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
    const float3 tangent = MediumUnit(MediumCross(helper, incoming)),
                 bitangent = MediumCross(incoming, tangent);
    const float tx = float(sine * cos(phi)), ty = float(sine * sin(phi));
    const float3 outgoing =
        MediumUnit(make_float3(incoming.x * float(c) + tangent.x * tx + bitangent.x * ty,
                               incoming.y * float(c) + tangent.y * tx + bitangent.y * ty,
                               incoming.z * float(c) + tangent.z * tx + bitangent.z * ty));
    pdf = importance ? MediumPhase(MediumDot(incoming, outgoing), anisotropy) : float(1 / MediumFourPi);
    return outgoing;
}
