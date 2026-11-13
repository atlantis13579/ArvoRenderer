// Independent native Metal geometric emission. PBRT 4e, "Triangle Meshes" and
// "Area Lights": https://pbr-book.org/4ed/Light_Sources/Area_Lights
// Triangle edges map barycentric coordinates; ellipsoid columns map a unit
// sphere. Its area Jacobian is |cofactor(B)*n|, hence pA=1/(4*pi*J(n)).
// Existing material emission is two-sided: pDir=abs(n.w)/(2*pi), flux=2*pi*L/pA.
constant uint SurfaceEmitterTriangle=0,SurfaceEmitterEllipsoid=1,SurfaceEmitterValid=1;
constant float SurfaceEmissionPi=3.14159265358979323846f;
constant float SurfaceSideProbability=0.5f,SurfaceParallelThreshold=0.999f;
struct SurfaceEmissionSource
{
    uint Kind;
    float3 Origin,AxisX,AxisY,AxisZ,Radiance;
};
struct SurfaceEmissionSample
{
    float3 Origin,Direction,Normal,Value,Weight;
    float PositionPdf,DirectionPdf;
    uint Flags;
};
struct SurfaceMetricData
{
    float3 X,Y,Z,CofactorX,CofactorY,CofactorZ;
    float Scale,Orientation;
};
float SurfacePeak(float3 value) { return max(fabs(value.x),max(fabs(value.y),fabs(value.z))); }
float SurfaceLength(float3 value)
{
    float scale=SurfacePeak(value);
    if (!(scale > 0)) return 0;
    float3 normalized=value/scale;
    return scale*sqrt(dot(normalized,normalized));
}
SurfaceMetricData SurfaceMetric(SurfaceEmissionSource source)
{
    SurfaceMetricData metric{};
    metric.Scale=max(SurfacePeak(source.AxisX),SurfacePeak(source.AxisY));
    if (source.Kind==SurfaceEmitterEllipsoid) metric.Scale=max(metric.Scale,SurfacePeak(source.AxisZ));
    if (!(metric.Scale > 0) || !isfinite(metric.Scale)) return SurfaceMetricData{};
    metric.X=source.AxisX; metric.Y=source.AxisY; metric.Z=source.AxisZ;
    metric.CofactorZ=cross(metric.X,metric.Y);
    if (source.Kind==SurfaceEmitterEllipsoid)
    {
        metric.CofactorX=cross(metric.Y,metric.Z); metric.CofactorY=cross(metric.Z,metric.X);
        float axisScale=SurfacePeak(metric.X),areaScale=SurfacePeak(metric.CofactorX);
        if (axisScale > 0 && areaScale > 0)
            metric.Orientation=dot(metric.X/axisScale,metric.CofactorX/areaScale);
    }
    return metric;
}
float3 SurfaceCofactor(SurfaceMetricData metric,float3 n)
{
    return metric.CofactorX*n.x+metric.CofactorY*n.y+metric.CofactorZ*n.z;
}
float SurfaceAreaDensity(SurfaceMetricData metric,float jacobian,float originalDensity)
{
    if (!(metric.Scale > 0 && jacobian > 0)) return 0;
    // World cofactors avoid subnormal determinants from a common matrix scale.
    float pdf=originalDensity/jacobian;
    return pdf > 0 && isfinite(pdf) ? pdf : 0;
}
// Query on the emitter with unit geometric normal and emitted direction. Recover
// sphere parameters through B^T*nWorld instead of subtracting world positions.
float2 SurfaceEmissionPdf(SurfaceEmissionSource source,float3 normal,float3 direction)
{
    SurfaceMetricData metric=SurfaceMetric(source);
    if (!(metric.Scale > 0)) return float2(0,0);
    float areaPdf=0;
    if (source.Kind==SurfaceEmitterTriangle)
    {
        float magnitude=SurfaceLength(metric.CofactorZ);
        areaPdf=SurfaceAreaDensity(metric,magnitude,2);
        if (!(areaPdf > 0)) return float2(0,0);
        normal=metric.CofactorZ/magnitude;
    }
    else if (source.Kind==SurfaceEmitterEllipsoid && metric.Orientation!=0)
    {
        float3 original=float3(dot(metric.X,normal),dot(metric.Y,normal),dot(metric.Z,normal));
        float magnitude=SurfaceLength(original);
        if (!(magnitude > 0)) return float2(0,0);
        areaPdf=SurfaceAreaDensity(metric,SurfaceLength(SurfaceCofactor(metric,original/magnitude)),1/(4*SurfaceEmissionPi));
    }
    if (!(areaPdf > 0)) return float2(0,0);
    return float2(areaPdf,SurfaceSideProbability*fabs(dot(normal,direction))/SurfaceEmissionPi);
}
SurfaceEmissionSample SampleSurfaceEmission(SurfaceEmissionSource source,float2 positionU,float2 directionU)
{
    SurfaceEmissionSample sample{};
    SurfaceMetricData metric=SurfaceMetric(source);
    if (!(metric.Scale > 0)) return sample;
    if (source.Kind==SurfaceEmitterTriangle)
    {
        float twiceArea=SurfaceLength(metric.CofactorZ);
        sample.PositionPdf=SurfaceAreaDensity(metric,twiceArea,2);
        if (!(sample.PositionPdf > 0)) return SurfaceEmissionSample{};
        sample.Normal=metric.CofactorZ/twiceArea;
        float root=sqrt(positionU.x);
        sample.Origin=source.Origin+source.AxisX*(root*(1-positionU.y))+source.AxisY*(root*positionU.y);
    }
    else if (source.Kind==SurfaceEmitterEllipsoid && metric.Orientation!=0)
    {
        float y=1-2*positionU.x,radius=sqrt(max(0.0f,1-y*y)),phi=2*SurfaceEmissionPi*positionU.y;
        float3 local=float3(radius*cos(phi),y,radius*sin(phi));
        float3 cofactor=SurfaceCofactor(metric,local);
        float jacobian=SurfaceLength(cofactor);
        sample.PositionPdf=SurfaceAreaDensity(metric,jacobian,1/(4*SurfaceEmissionPi));
        if (!(sample.PositionPdf > 0)) return SurfaceEmissionSample{};
        sample.Normal=(cofactor/jacobian)*(metric.Orientation < 0 ? -1.0f : 1.0f);
        sample.Origin=source.Origin+source.AxisX*local.x+source.AxisY*local.y+source.AxisZ*local.z;
    }
    else return sample;
    bool front=directionU.x < SurfaceSideProbability;
    float radialU=front ? directionU.x/SurfaceSideProbability : (directionU.x-SurfaceSideProbability)/SurfaceSideProbability;
    float cosine=sqrt(max(0.0f,1-radialU));
    if (!(cosine > 0)) return SurfaceEmissionSample{};
    float radius=sqrt(radialU),phi=2*SurfaceEmissionPi*directionU.y;
    float3 helper=fabs(sample.Normal.y) < SurfaceParallelThreshold ? float3(0,1,0) : float3(1,0,0);
    float3 tangent=normalize(cross(helper,sample.Normal)),bitangent=cross(sample.Normal,tangent);
    sample.Direction=tangent*(radius*cos(phi))+bitangent*(radius*sin(phi))+sample.Normal*(front ? cosine : -cosine);
    sample.DirectionPdf=SurfaceSideProbability*cosine/SurfaceEmissionPi;
    sample.Value=source.Radiance;
    sample.Weight=(source.Radiance/sample.PositionPdf)*(SurfaceEmissionPi/SurfaceSideProbability);
    sample.Flags=SurfaceEmitterValid;
    return sample;
}
