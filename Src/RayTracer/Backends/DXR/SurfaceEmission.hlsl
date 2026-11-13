// Independent DXR emitted surface endpoints. PBRT 4e, "Area Lights" and
// "Triangle Meshes". Two cosine hemispheres each have selection probability 1/2.
// For an affine sphere x=c+B*n, surface area stretches by |cofactor(B)*n|.
// Emission densities use that local Jacobian, including reflected/sheared B.
static const uint SurfaceEmitterTriangle=0,SurfaceEmitterEllipsoid=1,SurfaceEmitterValid=1;
static const float SurfaceEmissionPi=3.14159265358979323846f;
static const float SurfaceSideMass=0.5f,SurfaceFrameThreshold=0.999f;
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
    float3 ColumnX,ColumnY,ColumnZ,AreaX,AreaY,AreaZ;
    float Scale,Orientation;
};
SurfaceEmissionSample NullSurfaceEmission()
{
    SurfaceEmissionSample sample;
    sample.Origin=sample.Direction=sample.Normal=sample.Value=sample.Weight=float3(0,0,0);
    sample.PositionPdf=sample.DirectionPdf=0; sample.Flags=0;
    return sample;
}
float SurfaceMagnitude(float3 value) { return max(abs(value.x),max(abs(value.y),abs(value.z))); }
float SurfaceVectorLength(float3 v)
{
    float peak=SurfaceMagnitude(v);
    if (!(peak > 0)) return 0;
    float3 relative=v/peak;
    return peak*sqrt(dot(relative,relative));
}
SurfaceMetricData BuildSurfaceMetric(SurfaceEmissionSource source)
{
    SurfaceMetricData data;
    data.ColumnX=data.ColumnY=data.ColumnZ=data.AreaX=data.AreaY=data.AreaZ=float3(0,0,0);
    data.Orientation=0;
    data.Scale=max(SurfaceMagnitude(source.AxisX),SurfaceMagnitude(source.AxisY));
    if (source.Kind==SurfaceEmitterEllipsoid) data.Scale=max(data.Scale,SurfaceMagnitude(source.AxisZ));
    if (!(data.Scale > 0) || !isfinite(data.Scale)) { data.Scale=0; return data; }
    data.ColumnX=source.AxisX; data.ColumnY=source.AxisY; data.ColumnZ=source.AxisZ;
    data.AreaZ=cross(data.ColumnX,data.ColumnY);
    if (source.Kind==SurfaceEmitterEllipsoid)
    {
        data.AreaX=cross(data.ColumnY,data.ColumnZ); data.AreaY=cross(data.ColumnZ,data.ColumnX);
        float columnScale=SurfaceMagnitude(data.ColumnX),areaScale=SurfaceMagnitude(data.AreaX);
        if (columnScale > 0 && areaScale > 0)
            data.Orientation=dot(data.ColumnX/columnScale,data.AreaX/areaScale);
    }
    return data;
}
float3 ApplySurfaceCofactor(SurfaceMetricData data,float3 n)
{
    return data.AreaX*n.x+data.AreaY*n.y+data.AreaZ*n.z;
}
float SurfacePositionDensity(SurfaceMetricData data,float worldArea,float density)
{
    if (!(data.Scale > 0 && worldArea > 0)) return 0;
    // Cofactors retain world units; normalize only vectors used for sign/length.
    float result=density/worldArea;
    return result > 0 && isfinite(result) ? result : 0;
}
// Assumes a point on this source and a unit geometric normal/direction. A
// triangle uses its edge cross product; the sphere recovers n from B^T*nWorld.
float2 SurfaceEmissionPdf(SurfaceEmissionSource source,float3 geometricNormal,float3 direction)
{
    SurfaceMetricData data=BuildSurfaceMetric(source);
    if (!(data.Scale > 0)) return float2(0,0);
    float areaPdf=0;
    switch (source.Kind)
    {
    case SurfaceEmitterTriangle:
    {
        float crossLength=SurfaceVectorLength(data.AreaZ);
        areaPdf=SurfacePositionDensity(data,crossLength,2);
        if (!(areaPdf > 0)) return float2(0,0);
        geometricNormal=data.AreaZ/crossLength;
        break;
    }
    case SurfaceEmitterEllipsoid:
    {
        if (data.Orientation==0) return float2(0,0);
        float3 unitParameter=float3(dot(data.ColumnX,geometricNormal),dot(data.ColumnY,geometricNormal),dot(data.ColumnZ,geometricNormal));
        float length=SurfaceVectorLength(unitParameter);
        if (!(length > 0)) return float2(0,0);
        float jacobian=SurfaceVectorLength(ApplySurfaceCofactor(data,unitParameter/length));
        areaPdf=SurfacePositionDensity(data,jacobian,1/(4*SurfaceEmissionPi));
        break;
    }
    }
    if (!(areaPdf > 0)) return float2(0,0);
    return float2(areaPdf,SurfaceSideMass*abs(dot(geometricNormal,direction))/SurfaceEmissionPi);
}
SurfaceEmissionSample SampleSurfaceEmission(SurfaceEmissionSource source,float2 positionU,float2 directionU)
{
    SurfaceEmissionSample emitted=NullSurfaceEmission();
    SurfaceMetricData data=BuildSurfaceMetric(source);
    if (!(data.Scale > 0)) return emitted;
    if (source.Kind==SurfaceEmitterTriangle)
    {
        float crossLength=SurfaceVectorLength(data.AreaZ);
        emitted.PositionPdf=SurfacePositionDensity(data,crossLength,2);
        if (!(emitted.PositionPdf > 0)) return NullSurfaceEmission();
        emitted.Normal=data.AreaZ/crossLength;
        float root=sqrt(positionU.x),alongY=root*positionU.y,alongX=root*(1-positionU.y);
        emitted.Origin=source.Origin+source.AxisX*alongX+source.AxisY*alongY;
    }
    else if (source.Kind==SurfaceEmitterEllipsoid && data.Orientation!=0)
    {
        float vertical=1-2*positionU.x,ring=sqrt(max(0.0f,1-vertical*vertical)),angle=2*SurfaceEmissionPi*positionU.y;
        float3 n=float3(ring*cos(angle),vertical,ring*sin(angle));
        float3 transformed=ApplySurfaceCofactor(data,n);
        float jacobian=SurfaceVectorLength(transformed);
        emitted.PositionPdf=SurfacePositionDensity(data,jacobian,1/(4*SurfaceEmissionPi));
        if (!(emitted.PositionPdf > 0)) return NullSurfaceEmission();
        emitted.Normal=(transformed/jacobian)*(data.Orientation < 0 ? -1.0f : 1.0f);
        emitted.Origin=source.Origin+source.AxisX*n.x+source.AxisY*n.y+source.AxisZ*n.z;
    }
    else return emitted;
    bool positive=directionU.x < SurfaceSideMass;
    float diskRadiusSquared=positive ? directionU.x/SurfaceSideMass : (directionU.x-SurfaceSideMass)/SurfaceSideMass;
    float cosTheta=sqrt(max(0.0f,1-diskRadiusSquared));
    if (!(cosTheta > 0)) return NullSurfaceEmission();
    float ring=sqrt(diskRadiusSquared),angle=2*SurfaceEmissionPi*directionU.y;
    float3 axis=abs(emitted.Normal.y) < SurfaceFrameThreshold ? float3(0,1,0) : float3(1,0,0);
    float3 right=normalize(cross(axis,emitted.Normal)),up=cross(emitted.Normal,right);
    emitted.Direction=right*(ring*cos(angle))+up*(ring*sin(angle))+emitted.Normal*(positive ? cosTheta : -cosTheta);
    emitted.DirectionPdf=SurfaceSideMass*cosTheta/SurfaceEmissionPi;
    emitted.Value=source.Radiance;
    emitted.Weight=(source.Radiance/emitted.PositionPdf)*(SurfaceEmissionPi/SurfaceSideMass);
    emitted.Flags=SurfaceEmitterValid;
    return emitted;
}
