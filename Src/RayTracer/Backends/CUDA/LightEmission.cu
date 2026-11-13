#include <cuda_runtime.h>

namespace ArvoCudaEmission
{
// Independent CUDA emitted-ray sampling. PBRT 3e, section 16.1.2,
// "The Path-Space Measurement Equation". No camera-path estimator is reused.
// Strength = RGB * scene intensity. Rectangles divide it by emitting area;
// points interpret it as radiant intensity, distant lights as irradiance.
// Continuous PDF queries return zero in delta dimensions. Sample PDFs instead
// store unit masses, identified by Flags. All PDFs exclude emitter selection.
constexpr float Pi = 3.14159265358979323846f, ParallelThreshold = 0.999f;
constexpr float RedLuminance = 0.2126f, GreenLuminance = 0.7152f, BlueLuminance = 0.0722f;
enum Kind : unsigned { Directional, Point, Rectangle, Environment };
enum Flag : unsigned { Valid = 1, DeltaPosition = 2, DeltaDirection = 4, Infinite = 8 };
struct Domain { float3 Center; float Radius; };
struct Source
{
    unsigned Type;
    float3 Position, Direction, Strength, BackgroundTop, BackgroundBottom;
    float2 Size;
};
struct Sample
{
    float3 Origin, Direction, Normal, Value, Weight;
    float PositionPdf, DirectionPdf;
    unsigned Flags;
};
__device__ float3 Add(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ float3 Scale(float3 a, float s) { return make_float3(a.x*s, a.y*s, a.z*s); }
__device__ float Dot(float3 a, float3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ float3 Cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
__device__ float3 Unit(float3 v) { return Scale(v, 1/sqrtf(Dot(v,v))); }
__device__ void Basis(float3 n, float3& t, float3& b)
{
    float3 helper = fabsf(n.y) < ParallelThreshold ? make_float3(0,1,0) : make_float3(1,0,0);
    t = Unit(Cross(helper,n)); b = Cross(n,t);
}
__device__ float2 UniformDisk(float2 u)
{
    float radius = sqrtf(u.x), phi = 2*Pi*u.y;
    return make_float2(radius*cosf(phi),radius*sinf(phi));
}
__device__ float3 SphereLatitude(float y, float u)
{
    float r = sqrtf(fmaxf(0,1-y*y)), phi = 2*Pi*u;
    return make_float3(r*cosf(phi),y,r*sinf(phi));
}
__device__ float2 EnvironmentWeights(const Source& light)
{
    const float3 top = light.BackgroundTop, bottom = light.BackgroundBottom;
    float scale = fmaxf(fmaxf(top.x,top.y), fmaxf(top.z,fmaxf(bottom.x,fmaxf(bottom.y,bottom.z))));
    if (!(scale > 0)) return make_float2(0,0);
    // Normalize HDR before taking luminance, avoiding a bright top+bottom sum.
    float lo = Dot(make_float3(bottom.x/scale,bottom.y/scale,bottom.z/scale),make_float3(RedLuminance,GreenLuminance,BlueLuminance));
    float hi = Dot(make_float3(top.x/scale,top.y/scale,top.z/scale),make_float3(RedLuminance,GreenLuminance,BlueLuminance));
    return make_float2(lo/(lo+hi),hi/(lo+hi));
}
__device__ float EnvironmentPdf(float2 weights, float outwardY)
{
    return ((1-outwardY)*weights.x+(1+outwardY)*weights.y)/(4*Pi);
}
// Position is already known to lie on the source support. Direction is unit.
__device__ float2 Pdf(const Source& light, const Domain& domain, float3 direction)
{
    if (light.Type == Point) return make_float2(0,1/(4*Pi));
    if (light.Type == Rectangle)
    {
        if (!(light.Size.x > 0 && light.Size.y > 0)) return make_float2(0,0);
        return make_float2(1/(light.Size.x*light.Size.y),fmaxf(0,Dot(Unit(light.Direction),direction))/Pi);
    }
    if (!(domain.Radius > 0)) return make_float2(0,0);
    float areaPdf = 1/(Pi*domain.Radius*domain.Radius);
    if (light.Type == Directional) return make_float2(areaPdf,0);
    if (light.Type == Environment) return make_float2(areaPdf,EnvironmentPdf(EnvironmentWeights(light),-direction.y));
    return make_float2(0,0);
}
// Source is validated: finite/nonnegative RGB, nonzero axes, unit-square samples.
// Infinite sources additionally require a positive enclosing scene sphere.
__device__ Sample Emit(const Source& light, const Domain& domain, float2 positionU, float2 directionU)
{
    Sample result{};
    if (light.Type == Point)
    {
        result.Origin = light.Position;
        result.Direction = SphereLatitude(1-2*directionU.x,directionU.y);
        result.Normal = result.Direction;
        result.Value = light.Strength;
        result.Weight = Scale(light.Strength,4*Pi);
        result.PositionPdf = 1; result.DirectionPdf = 1/(4*Pi);
        result.Flags = Valid|DeltaPosition;
        return result;
    }
    if (light.Type == Rectangle)
    {
        if (!(light.Size.x > 0 && light.Size.y > 0)) return result;
        result.Normal = Unit(light.Direction);
        float3 right,up; Basis(result.Normal,right,up);
        result.Origin = Add(light.Position,Add(Scale(right,(positionU.x-0.5f)*light.Size.x),
                                             Scale(up,(positionU.y-0.5f)*light.Size.y)));
        float2 disk = UniformDisk(directionU);
        float cosine = sqrtf(fmaxf(0,1-directionU.x));
        if (!(cosine > 0)) return Sample{};
        result.Direction = Add(Add(Scale(right,disk.x),Scale(up,disk.y)),Scale(result.Normal,cosine));
        result.PositionPdf = 1/(light.Size.x*light.Size.y); result.DirectionPdf = cosine/Pi;
        result.Value = Scale(light.Strength,result.PositionPdf);
        result.Weight = Scale(light.Strength,Pi);
        result.Flags = Valid;
        return result;
    }
    if (!(domain.Radius > 0)) return result;
    float3 outward;
    if (light.Type == Directional)
    {
        outward = Unit(light.Direction);
        result.Value = light.Strength; result.DirectionPdf = 1;
        result.Flags = Valid|Infinite|DeltaDirection;
    }
    else if (light.Type == Environment)
    {
        float2 weights = EnvironmentWeights(light);
        if (!(weights.x+weights.y > 0)) return result;
        float u = directionU.x;
        float denominator = weights.x + sqrtf((1-u)*weights.x*weights.x + u*weights.y*weights.y);
        float q = denominator > 0 ? fminf(1,fmaxf(0,u/denominator)) : 0;
        outward = SphereLatitude(2*q-1,directionU.y);
        result.DirectionPdf = EnvironmentPdf(weights,outward.y);
        if (!(result.DirectionPdf > 0)) return Sample{};
        result.Value = Add(Scale(light.BackgroundBottom,1-q),Scale(light.BackgroundTop,q));
        result.Flags = Valid|Infinite;
    }
    else return result;
    float3 right,up; Basis(outward,right,up);
    float2 disk = UniformDisk(positionU);
    result.Origin = Add(domain.Center,Scale(Add(outward,Add(Scale(right,disk.x),Scale(up,disk.y))),domain.Radius));
    result.Direction = Scale(outward,-1); result.Normal = result.Direction;
    result.PositionPdf = 1/(Pi*domain.Radius*domain.Radius);
    result.Weight = Scale(result.Value,1/(result.PositionPdf*result.DirectionPdf));
    return result;
}
} // namespace ArvoCudaEmission
