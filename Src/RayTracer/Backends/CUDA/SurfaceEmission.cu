#include <cuda_runtime.h>
namespace ArvoCudaSurfaceEmission
{
// Independent CUDA geometric emission; PBRT 4e, "Triangle Meshes" / "Area Lights".
// Triangle axes are edges. Ellipsoid axes are columns mapping a unit sphere.
// The sphere's uniform parameter samples require pA=1/(4*pi*|cofactor(B)*n|).
// Two-sided cosine emission contributes a hemisphere PMF of 1/2, so conditional
// flux weight is 2*pi*Radiance/pA. No other backend/estimator supplies arithmetic.
constexpr unsigned Triangle=0,Ellipsoid=1,Valid=1;
constexpr float Pi=3.14159265358979323846f,SideProbability=0.5f,ParallelThreshold=0.999f;
struct Source { unsigned Kind; float3 Origin,AxisX,AxisY,AxisZ,Radiance; };
struct Sample
{
    float3 Origin,Direction,Normal,Value,Weight;
    float PositionPdf,DirectionPdf;
    unsigned Flags;
};
struct Metric
{
    float3 X,Y,Z,CofactorX,CofactorY,CofactorZ;
    float Scale,Orientation;
};
__device__ float3 Add(float3 a,float3 b) { return make_float3(a.x+b.x,a.y+b.y,a.z+b.z); }
__device__ float3 Times(float3 v,float x) { return make_float3(v.x*x,v.y*x,v.z*x); }
__device__ float3 Divide(float3 v,float x) { return make_float3(v.x/x,v.y/x,v.z/x); }
__device__ float Dot(float3 a,float3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ float3 Cross(float3 a,float3 b)
{ return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x); }
__device__ float Peak(float3 v) { return fmaxf(fabsf(v.x),fmaxf(fabsf(v.y),fabsf(v.z))); }
__device__ float Length(float3 v)
{
    float scale=Peak(v);
    if (!(scale > 0)) return 0;
    v=Divide(v,scale);
    return scale*sqrtf(Dot(v,v));
}
__device__ Metric ShapeMetric(Source source)
{
    Metric metric{};
    metric.Scale=fmaxf(Peak(source.AxisX),Peak(source.AxisY));
    if (source.Kind==Ellipsoid) metric.Scale=fmaxf(metric.Scale,Peak(source.AxisZ));
    if (!(metric.Scale > 0) || !isfinite(metric.Scale)) return Metric{};
    metric.X=source.AxisX; metric.Y=source.AxisY; metric.Z=source.AxisZ;
    metric.CofactorZ=Cross(metric.X,metric.Y);
    if (source.Kind==Ellipsoid)
    {
        metric.CofactorX=Cross(metric.Y,metric.Z); metric.CofactorY=Cross(metric.Z,metric.X);
        float columnScale=Peak(metric.X),areaScale=Peak(metric.CofactorX);
        if (columnScale > 0 && areaScale > 0)
            metric.Orientation=Dot(Divide(metric.X,columnScale),Divide(metric.CofactorX,areaScale));
    }
    return metric;
}
__device__ float3 TransformCofactor(Metric m,float3 n)
{ return Add(Add(Times(m.CofactorX,n.x),Times(m.CofactorY,n.y)),Times(m.CofactorZ,n.z)); }
__device__ float AreaDensity(Metric m,float jacobian,float reference)
{
    if (!(m.Scale > 0 && jacobian > 0)) return 0;
    // Normalize sign/length calculations, not the matrix's physical area scale.
    float density=reference/jacobian;
    return density > 0 && isfinite(density) ? density : 0;
}
// The query lies on the surface with a unit geometric normal and ray direction.
__device__ float2 Pdf(Source source,float3 normal,float3 direction)
{
    Metric metric=ShapeMetric(source);
    if (!(metric.Scale > 0)) return make_float2(0,0);
    float position=0;
    if (source.Kind==Triangle)
    {
        float twiceArea=Length(metric.CofactorZ);
        position=AreaDensity(metric,twiceArea,2);
        if (!(position > 0)) return make_float2(0,0);
        normal=Divide(metric.CofactorZ,twiceArea);
    }
    else if (source.Kind==Ellipsoid && metric.Orientation!=0)
    {
        float3 local=make_float3(Dot(metric.X,normal),Dot(metric.Y,normal),Dot(metric.Z,normal));
        float length=Length(local);
        if (!(length > 0)) return make_float2(0,0);
        position=AreaDensity(metric,Length(TransformCofactor(metric,Divide(local,length))),1/(4*Pi));
    }
    if (!(position > 0)) return make_float2(0,0);
    return make_float2(position,SideProbability*fabsf(Dot(normal,direction))/Pi);
}
__device__ Sample Emit(Source source,float2 positionU,float2 directionU)
{
    Sample sample{};
    Metric metric=ShapeMetric(source);
    if (!(metric.Scale > 0)) return sample;
    if (source.Kind==Triangle)
    {
        float twiceArea=Length(metric.CofactorZ);
        sample.PositionPdf=AreaDensity(metric,twiceArea,2);
        if (!(sample.PositionPdf > 0)) return Sample{};
        sample.Normal=Divide(metric.CofactorZ,twiceArea);
        float root=sqrtf(positionU.x);
        sample.Origin=Add(source.Origin,Add(Times(source.AxisX,root*(1-positionU.y)),Times(source.AxisY,root*positionU.y)));
    }
    else if (source.Kind==Ellipsoid && metric.Orientation!=0)
    {
        float y=1-2*positionU.x,radius=sqrtf(fmaxf(0,1-y*y)),phi=2*Pi*positionU.y;
        float3 local=make_float3(radius*cosf(phi),y,radius*sinf(phi));
        float3 cofactor=TransformCofactor(metric,local);
        float jacobian=Length(cofactor);
        sample.PositionPdf=AreaDensity(metric,jacobian,1/(4*Pi));
        if (!(sample.PositionPdf > 0)) return Sample{};
        sample.Normal=Times(Divide(cofactor,jacobian),metric.Orientation < 0 ? -1.0f : 1.0f);
        sample.Origin=Add(source.Origin,Add(Add(Times(source.AxisX,local.x),Times(source.AxisY,local.y)),Times(source.AxisZ,local.z)));
    }
    else return sample;
    bool front=directionU.x < SideProbability;
    float radialU=front ? directionU.x/SideProbability : (directionU.x-SideProbability)/SideProbability;
    float cosine=sqrtf(fmaxf(0,1-radialU));
    if (!(cosine > 0)) return Sample{};
    float radius=sqrtf(radialU),phi=2*Pi*directionU.y;
    float3 reference=fabsf(sample.Normal.y) < ParallelThreshold ? make_float3(0,1,0) : make_float3(1,0,0);
    float3 tangent=Cross(reference,sample.Normal); tangent=Divide(tangent,Length(tangent));
    float3 bitangent=Cross(sample.Normal,tangent);
    sample.Direction=Add(Add(Times(tangent,radius*cosf(phi)),Times(bitangent,radius*sinf(phi))),Times(sample.Normal,front ? cosine : -cosine));
    sample.DirectionPdf=SideProbability*cosine/Pi;
    sample.Value=source.Radiance;
    sample.Weight=Times(Divide(source.Radiance,sample.PositionPdf),Pi/SideProbability);
    sample.Flags=Valid;
    return sample;
}
} // namespace ArvoCudaSurfaceEmission
