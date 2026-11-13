// Independent compute surface emission. PBRT 4e, "Triangle Meshes" and
// "Area Lights". Geometric material emission is two-sided in existing demos.
// Uniform unit-sphere samples become ellipsoid samples with area Jacobian
// J(n)=|cofactor(B)*n|; pA=1/(4*pi*J). Constant sphere area PDFs are incorrect
// after shear/nonuniform scale. Columns B already include the sphere radius.
static const uint SurfaceEmitterTriangle=0, SurfaceEmitterEllipsoid=1, SurfaceEmitterValid=1;
static const float SurfaceEmissionPi=3.14159265358979323846f;
static const float SurfaceEmissionSideProbability=0.5f, SurfaceEmissionAxisThreshold=0.999f;
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
struct SurfaceEmissionMetric
{
    float3 X,Y,Z,CofactorX,CofactorY,CofactorZ;
    float Scale,Orientation;
};
SurfaceEmissionSample EmptySurfaceEmission()
{
    SurfaceEmissionSample value;
    value.Origin=value.Direction=value.Normal=value.Value=value.Weight=float3(0,0,0);
    value.PositionPdf=value.DirectionPdf=0; value.Flags=0;
    return value;
}
float SurfaceEmissionPeak(float3 v) { return max(abs(v.x),max(abs(v.y),abs(v.z))); }
float SurfaceEmissionLength(float3 v)
{
    float scale=SurfaceEmissionPeak(v);
    if (!(scale > 0)) return 0;
    float3 unit=v/scale;
    return scale*sqrt(dot(unit,unit));
}
SurfaceEmissionMetric SurfaceMetric(SurfaceEmissionSource source)
{
    SurfaceEmissionMetric metric;
    metric.X=metric.Y=metric.Z=metric.CofactorX=metric.CofactorY=metric.CofactorZ=float3(0,0,0);
    metric.Orientation=0;
    metric.Scale=max(SurfaceEmissionPeak(source.AxisX),SurfaceEmissionPeak(source.AxisY));
    if (source.Kind==SurfaceEmitterEllipsoid) metric.Scale=max(metric.Scale,SurfaceEmissionPeak(source.AxisZ));
    if (!(metric.Scale > 0) || !isfinite(metric.Scale)) { metric.Scale=0; return metric; }
    metric.X=source.AxisX; metric.Y=source.AxisY; metric.Z=source.AxisZ;
    metric.CofactorZ=cross(metric.X,metric.Y);
    if (source.Kind==SurfaceEmitterEllipsoid)
    {
        metric.CofactorX=cross(metric.Y,metric.Z); metric.CofactorY=cross(metric.Z,metric.X);
        float axisScale=SurfaceEmissionPeak(metric.X),areaScale=SurfaceEmissionPeak(metric.CofactorX);
        if (axisScale > 0 && areaScale > 0)
            metric.Orientation=dot(metric.X/axisScale,metric.CofactorX/areaScale);
    }
    return metric;
}
float3 SurfaceCofactor(SurfaceEmissionMetric metric,float3 normal)
{
    return metric.CofactorX*normal.x+metric.CofactorY*normal.y+metric.CofactorZ*normal.z;
}
float SurfaceAreaPdf(SurfaceEmissionMetric metric,float jacobian,float referenceDensity)
{
    if (!(metric.Scale > 0 && jacobian > 0)) return 0;
    // Keep world-area products; common-scale normalization loses very thin axes.
    float density=referenceDensity/jacobian;
    return density > 0 && isfinite(density) ? density : 0;
}
// Query is on the emitter; normal/direction are unit world-space geometric vectors.
// Triangle axes are its two edges. No shading normal or source-selection PMF enters here.
float2 SurfaceEmissionPdf(SurfaceEmissionSource source,float3 normal,float3 direction)
{
    SurfaceEmissionMetric metric=SurfaceMetric(source);
    if (!(metric.Scale > 0)) return float2(0,0);
    float areaPdf=0;
    if (source.Kind==SurfaceEmitterTriangle)
    {
        float magnitude=SurfaceEmissionLength(metric.CofactorZ);
        areaPdf=SurfaceAreaPdf(metric,magnitude,2);
        if (!(areaPdf > 0)) return float2(0,0);
        normal=metric.CofactorZ/magnitude;
    }
    else if (source.Kind==SurfaceEmitterEllipsoid && metric.Orientation!=0)
    {
        float3 objectNormal=float3(dot(metric.X,normal),dot(metric.Y,normal),dot(metric.Z,normal));
        float magnitude=SurfaceEmissionLength(objectNormal);
        if (!(magnitude > 0)) return float2(0,0);
        float jacobian=SurfaceEmissionLength(SurfaceCofactor(metric,objectNormal/magnitude));
        areaPdf=SurfaceAreaPdf(metric,jacobian,1/(4*SurfaceEmissionPi));
    }
    if (!(areaPdf > 0)) return float2(0,0);
    return float2(areaPdf,SurfaceEmissionSideProbability*abs(dot(normal,direction))/SurfaceEmissionPi);
}
SurfaceEmissionSample SampleSurfaceEmission(SurfaceEmissionSource source,float2 positionU,float2 directionU)
{
    SurfaceEmissionSample result=EmptySurfaceEmission();
    SurfaceEmissionMetric metric=SurfaceMetric(source);
    if (!(metric.Scale > 0)) return result;
    if (source.Kind==SurfaceEmitterTriangle)
    {
        float twiceArea=SurfaceEmissionLength(metric.CofactorZ);
        result.PositionPdf=SurfaceAreaPdf(metric,twiceArea,2);
        if (!(result.PositionPdf > 0)) return EmptySurfaceEmission();
        result.Normal=metric.CofactorZ/twiceArea;
        float root=sqrt(positionU.x);
        result.Origin=source.Origin+source.AxisX*(root*(1-positionU.y))+source.AxisY*(root*positionU.y);
    }
    else if (source.Kind==SurfaceEmitterEllipsoid && metric.Orientation!=0)
    {
        float y=1-2*positionU.x, circle=sqrt(max(0.0f,1-y*y)),phi=2*SurfaceEmissionPi*positionU.y;
        float3 local=float3(circle*cos(phi),y,circle*sin(phi));
        float3 transformed=SurfaceCofactor(metric,local);
        float jacobian=SurfaceEmissionLength(transformed);
        result.PositionPdf=SurfaceAreaPdf(metric,jacobian,1/(4*SurfaceEmissionPi));
        if (!(result.PositionPdf > 0)) return EmptySurfaceEmission();
        result.Normal=(transformed/jacobian)*(metric.Orientation < 0 ? -1.0f : 1.0f);
        result.Origin=source.Origin+source.AxisX*local.x+source.AxisY*local.y+source.AxisZ*local.z;
    }
    else return result;
    bool front=directionU.x < SurfaceEmissionSideProbability;
    float radialU=front ? directionU.x/SurfaceEmissionSideProbability : (directionU.x-SurfaceEmissionSideProbability)/SurfaceEmissionSideProbability;
    float cosine=sqrt(max(0.0f,1-radialU));
    if (!(cosine > 0)) return EmptySurfaceEmission();
    float radius=sqrt(radialU),phi=2*SurfaceEmissionPi*directionU.y;
    float3 helper=abs(result.Normal.y) < SurfaceEmissionAxisThreshold ? float3(0,1,0) : float3(1,0,0);
    float3 tangent=normalize(cross(helper,result.Normal)),bitangent=cross(result.Normal,tangent);
    result.Direction=tangent*(radius*cos(phi))+bitangent*(radius*sin(phi))+result.Normal*(front ? cosine : -cosine);
    result.DirectionPdf=SurfaceEmissionSideProbability*cosine/SurfaceEmissionPi;
    result.Value=source.Radiance;
    result.Weight=(source.Radiance/result.PositionPdf)*(SurfaceEmissionPi/SurfaceEmissionSideProbability);
    result.Flags=SurfaceEmitterValid;
    return result;
}
