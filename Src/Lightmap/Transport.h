#pragma once
#include "Lightmap/Exchange.h"
#include <cmath>
#ifdef __CUDACC__
#define ARVO_LM_HD __host__ __device__
#else
#define ARVO_LM_HD
#endif
namespace Arvo::Lightmap::Transport
{
using namespace ArvoLightmap;
struct V
{
    float X=0,Y=0,Z=0;
    ARVO_LM_HD V():X(0),Y(0),Z(0) {}
    ARVO_LM_HD V(float x,float y,float z):X(x),Y(y),Z(z) {}
};
ARVO_LM_HD inline V operator+(V a,V b) { return {a.X+b.X,a.Y+b.Y,a.Z+b.Z}; }
ARVO_LM_HD inline V operator-(V a,V b) { return {a.X-b.X,a.Y-b.Y,a.Z-b.Z}; }
ARVO_LM_HD inline V operator-(V a) { return {-a.X,-a.Y,-a.Z}; }
ARVO_LM_HD inline V operator*(V a,float b) { return {a.X*b,a.Y*b,a.Z*b}; }
ARVO_LM_HD inline V operator/(V a,float b) { return {a.X/b,a.Y/b,a.Z/b}; }
ARVO_LM_HD inline float Dot(V a,V b) { return a.X*b.X+a.Y*b.Y+a.Z*b.Z; }
ARVO_LM_HD inline V Cross(V a,V b) { return {a.Y*b.Z-a.Z*b.Y,a.Z*b.X-a.X*b.Z,a.X*b.Y-a.Y*b.X}; }
template<class T> ARVO_LM_HD inline T Min(T a,T b) { return a<b?a:b; }
template<class T> ARVO_LM_HD inline T Max(T a,T b) { return a>b?a:b; }
template<class T> ARVO_LM_HD inline T Clamp(T a,T lo,T hi) { return Min(Max(a,lo),hi); }
template<class T> struct Span
{
    const T* Data=nullptr; uint32_t Count=0;
    ARVO_LM_HD const T& operator[](size_t i) const { return Data[i]; }
    ARVO_LM_HD const T* begin() const { return Data; }
    ARVO_LM_HD const T* end() const { return Data+Count; }
};
struct Image { uint32_t Width=0,Height=0; const uint8_t* Pixels=nullptr; };
struct Node;
struct Scene
{
    Settings Options;
    Span<Image> Textures;
    Span<Material> Materials;
    Span<Triangle> Triangles;
    uint32_t EnvironmentSide=0;
    Span<Vec3> EnvironmentFaces;
    Span<Light> Lights;
    Span<Receiver> Receivers;
    Span<Node> Nodes;
    Span<uint32_t> Order;
};
constexpr float Pi=3.14159265358979323846f;
ARVO_LM_HD inline V Vector(Vec3 v) { return {v.X,v.Y,v.Z}; }
ARVO_LM_HD inline V RGB(Vec4 v) { return {v.X,v.Y,v.Z}; }
ARVO_LM_HD inline V Mul(V a,V b) { return {a.X*b.X,a.Y*b.Y,a.Z*b.Z}; }
ARVO_LM_HD inline V Unit(V v,V fallback={0,1,0}) { const float l=Dot(v,v); return l>1e-20f? v/sqrtf(l):fallback; }
ARVO_LM_HD inline float Max(V v) { return Max(Max(v.X,v.Y),v.Z); }
ARVO_LM_HD inline bool Finite(V v) {
#ifdef __CUDA_ARCH__
    return isfinite(v.X)&&isfinite(v.Y)&&isfinite(v.Z);
#else
    return std::isfinite(v.X)&&std::isfinite(v.Y)&&std::isfinite(v.Z);
#endif
}
struct Random
{
    uint64_t State;
    ARVO_LM_HD float Next()
    {
        State+=0x9e3779b97f4a7c15ull;
        uint64_t z=State; z=(z^(z>>30))*0xbf58476d1ce4e5b9ull; z=(z^(z>>27))*0x94d049bb133111ebull;
        return float((z^(z>>31))>>40)*(1.0f/16777216.0f);
    }
};
ARVO_LM_HD inline V Hemisphere(V n,Random& random)
{
    const float u=random.Next(),phi=2*Pi*random.Next(),r=sqrtf(u);
    const V t=Unit(Cross(fabsf(n.Y)<0.999f?V(0,1,0):V(1,0,0),n)),b=Cross(n,t);
    return Unit(t*(r*cosf(phi))+b*(r*sinf(phi))+n*sqrtf(1-u));
}
ARVO_LM_HD inline float Decode(float x) { return x<=0.04045f?x/12.92f:powf((x+0.055f)/1.055f,2.4f); }
ARVO_LM_HD inline Vec4 Sample(const Scene& job,uint32_t id,Vec2 uv,bool srgb)
{
    if(id==Missing) return {1,1,1,1};
    const auto& image=job.Textures[id];
    const double x=(double(uv.X)-floor(double(uv.X)))*image.Width-0.5;
    const double y=(-double(uv.Y)-floor(-double(uv.Y)))*image.Height-0.5;
    const auto ix=int64_t(floor(x)),iy=int64_t(floor(y));
    auto read=[&](int64_t px,int64_t py) {
        px=(px%image.Width+image.Width)%image.Width; py=(py%image.Height+image.Height)%image.Height;
        const auto at=(size_t(py)*image.Width+px)*4;
        Vec4 c{image.Pixels[at]/255.f,image.Pixels[at+1]/255.f,image.Pixels[at+2]/255.f,image.Pixels[at+3]/255.f};
        if(srgb) { c.X=Decode(c.X); c.Y=Decode(c.Y); c.Z=Decode(c.Z); } return c;
    };
    auto mix=[](Vec4 a,Vec4 b,float f) { return Vec4{a.X+(b.X-a.X)*f,a.Y+(b.Y-a.Y)*f,a.Z+(b.Z-a.Z)*f,a.W+(b.W-a.W)*f}; };
    return mix(mix(read(ix,iy),read(ix+1,iy),float(x-ix)),mix(read(ix,iy+1),read(ix+1,iy+1),float(x-ix)),float(y-iy));
}
ARVO_LM_HD inline V Environment(const Scene& job,V direction)
{
    if(!job.EnvironmentSide) return Vector(job.Options.Environment);
    const V d{direction.X,direction.Y,-direction.Z};
    const float ax=fabsf(d.X),ay=fabsf(d.Y),az=fabsf(d.Z);
    uint32_t face; float u,v,major;
    if(ax>=ay && ax>=az) { face=d.X>=0?0u:1u; u=d.X>=0?-d.Z:d.Z; v=-d.Y; major=ax; }
    else if(ay>=az) { face=d.Y>=0?2u:3u; u=d.X; v=d.Y>=0?d.Z:-d.Z; major=ay; }
    else { face=d.Z>=0?4u:5u; u=d.Z>=0?d.X:-d.X; v=-d.Y; major=az; }
    const auto side=job.EnvironmentSide;
    const float x=(u/major+1)*0.5f*side-0.5f,y=(v/major+1)*0.5f*side-0.5f;
    const int ix=int(floor(x)),iy=int(floor(y));
    auto read=[&](int px,int py) { return Vector(job.EnvironmentFaces[(size_t(face)*side+Clamp(py,0,int(side)-1))*side+Clamp(px,0,int(side)-1)]); };
    const float fx=x-ix,fy=y-iy;
    return (read(ix,iy)*(1-fx)+read(ix+1,iy)*fx)*(1-fy)+(read(ix,iy+1)*(1-fx)+read(ix+1,iy+1)*fx)*fy;
}
struct Surface { V Position,Normal,Geometric,Albedo,Emission; bool Accepted=true; };
ARVO_LM_HD inline Surface Evaluate(const Scene& job,const Triangle& triangle,V weights,bool back=false)
{
    const auto* corners=reinterpret_cast<const Vertex*>(&triangle.Corners);
    const auto& a=corners[0]; const auto& b=corners[1]; const auto& c=corners[2];
    const auto& m=job.Materials[triangle.Material];
    const Vec2 uv{a.UV.X*weights.X+b.UV.X*weights.Y+c.UV.X*weights.Z,a.UV.Y*weights.X+b.UV.Y*weights.Y+c.UV.Y*weights.Z};
    const auto texture=Sample(job,m.Diffuse,uv,true);
    V color=RGB(a.Color)*weights.X+RGB(b.Color)*weights.Y+RGB(c.Color)*weights.Z;
    const float alpha=(a.Color.W*weights.X+b.Color.W*weights.Y+c.Color.W*weights.Z)*m.Base.W*texture.W*Sample(job,m.Opacity,uv,false).X;
    Surface s; s.Accepted=alpha>=m.AlphaCutoff;
    const V e1=Vector(b.Position)-Vector(a.Position),e2=Vector(c.Position)-Vector(a.Position);
    s.Position=Vector(a.Position)*weights.X+Vector(b.Position)*weights.Y+Vector(c.Position)*weights.Z;
    s.Geometric=Unit(Cross(e1,e2))*(back?-1.f:1.f);
    s.Normal=Unit(Vector(a.Normal)*weights.X+Vector(b.Normal)*weights.Y+Vector(c.Normal)*weights.Z,s.Geometric)*(back?-1.f:1.f);
    if(Dot(s.Normal,s.Geometric)<0) s.Normal=-s.Normal;
    if(m.Normal!=Missing)
    {
        const Vec2 d1{b.UV.X-a.UV.X,a.UV.Y-b.UV.Y},d2{c.UV.X-a.UV.X,a.UV.Y-c.UV.Y};
        const float determinant=d1.X*d2.Y-d1.Y*d2.X;
        if(fabsf(determinant)>1e-8f)
        {
            V t=(e1*d2.Y-e2*d1.Y)/determinant,bt=(e2*d1.X-e1*d2.X)/determinant;
            t=Unit(t-s.Normal*Dot(t,s.Normal)); bt=Cross(s.Normal,t)*(Dot(Cross(s.Normal,t),bt)<0?-1.f:1.f);
            const auto map=Sample(job,m.Normal,uv,false);
            const V mapped=Unit(t*(2*map.X-1)+bt*(2*map.Y-1)+s.Normal*(2*map.Z-1),s.Normal);
            if(Dot(mapped,s.Geometric)>0) s.Normal=mapped;
        }
    }
    const float metallic=Clamp(m.Metallic*Sample(job,m.MetallicMap,uv,false).Z,0.f,1.f);
    color=Mul(Mul(color,RGB(m.Base)),RGB(texture))*(m.DiffuseScale*(1-metallic));
    s.Albedo={Clamp(color.X,0.f,1.f),Clamp(color.Y,0.f,1.f),Clamp(color.Z,0.f,1.f)};
    s.Emission=Vector(m.Emission); return s;
}
struct Bounds
{
    V Lo={1e30f,1e30f,1e30f},Hi={-1e30f,-1e30f,-1e30f};
    ARVO_LM_HD void Add(V p) { Lo={Min(Lo.X,p.X),Min(Lo.Y,p.Y),Min(Lo.Z,p.Z)}; Hi={Max(Hi.X,p.X),Max(Hi.Y,p.Y),Max(Hi.Z,p.Z)}; }
    ARVO_LM_HD bool Hit(V o,V d,float near,float far) const
    {
        const float origin[]={o.X,o.Y,o.Z},dir[]={d.X,d.Y,d.Z},lo[]={Lo.X,Lo.Y,Lo.Z},hi[]={Hi.X,Hi.Y,Hi.Z};
        for(int i=0;i<3;++i) { if(dir[i]==0) { if(origin[i]<lo[i]||origin[i]>hi[i]) return false; }
            else { float a=(lo[i]-origin[i])/dir[i],b=(hi[i]-origin[i])/dir[i]; if(a>b) { const float temp=a; a=b; b=temp; }; near=Max(near,a); far=Min(far,b); if(far<near) return false; } }
        return true;
    }
};
struct Node { Bounds Box; uint32_t Begin=0,Count=0,Left=0,Right=0; };
// Rejected alpha hits never shorten the traversal interval.
ARVO_LM_HD inline bool Trace(const Scene& scene,V o,V d,float near,float far,Surface& surface,bool shadow=false)
{
    if(!scene.Nodes.Count) return false;
    uint32_t stack[64],count=1,closest=Missing; stack[0]=0; bool found=false;
    float distance=far;
    while(count)
    {
        const auto& node=scene.Nodes[stack[--count]];
        if(!node.Box.Hit(o,d,near,distance)) continue;
        if(!node.Count) { stack[count++]=node.Left; stack[count++]=node.Right; continue; }
        for(uint32_t i=0;i<node.Count;++i)
        {
            const auto id=scene.Order[node.Begin+i]; const auto& t=scene.Triangles[id];
            if(shadow && !t.CastShadow) continue;
            const auto* corners=reinterpret_cast<const Vertex*>(&t.Corners);
            const V a=Vector(corners[0].Position),e1=Vector(corners[1].Position)-a,e2=Vector(corners[2].Position)-a;
            const V p=Cross(d,e2); const float determinant=Dot(e1,p);
            if(fabsf(determinant)<1e-8f) continue;
            const float inverse=1/determinant; const V offset=o-a;
            const float u=Dot(offset,p)*inverse; if(u<0 || u>1) continue;
            const V q=Cross(offset,e1); const float v=Dot(d,q)*inverse; if(v<0 || u+v>1) continue;
            const float candidate=Dot(e2,q)*inverse;
            if(!(candidate>near && candidate<far) || candidate>distance || (candidate==distance && id>=closest)) continue;
            const auto material=Evaluate(scene,t,{1-u-v,u,v},determinant<0);
            if(!material.Accepted) continue;
            if(shadow) return true;
            distance=candidate; closest=id; surface=material; found=true;
        }
    }
    return found;
}
ARVO_LM_HD inline V Direct(const Scene& job,const Surface& s,Random& random)
{
    V result;
    for(const auto& light:job.Lights)
    {
        V direction,endpoint=Vector(light.Position); float weight=1,distance=1e30f;
        if(light.Type==0) direction=Unit(Vector(light.Direction));
        else
        {
            V emitterNormal;
            if(light.Type==2)
            {
                emitterNormal=Unit(Vector(light.Direction));
                const V tangent=Unit(Cross(fabsf(emitterNormal.Y)<0.999f?V(0,1,0):V(1,0,0),emitterNormal));
                const float u=random.Next(),v=random.Next();
                endpoint=endpoint+tangent*((u-0.5f)*light.Size.X)+Cross(emitterNormal,tangent)*((v-0.5f)*light.Size.Y);
            }
            const V offset=endpoint-s.Position; const float squared=Dot(offset,offset);
            if(squared<=1e-20f) continue;
            distance=sqrtf(squared); direction=offset/distance; weight=1/squared;
            if(light.Range>0) { const float r=squared/(light.Range*light.Range); const float window=Max(1-r*r,0.f); weight*=window*window; }
            if(light.Type==2) weight*=light.Size.X*light.Size.Y*Max(0.f,Dot(emitterNormal,-direction));
        }
        const float cosine=Max(0.f,Dot(s.Normal,direction));
        if(cosine==0 || Dot(s.Geometric,direction)<=0 || weight==0) continue;
        if(light.CastShadow)
        {
            const V origin=s.Position+s.Geometric*job.Options.Bias;
            V shadowDirection=direction; float limit=distance;
            if(light.Type!=0) { const V segment=endpoint-origin; limit=sqrtf(Dot(segment,segment)); shadowDirection=segment/limit; }
            Surface blocker;
            if(Trace(job,origin,shadowDirection,job.Options.Bias*0.1f,limit-job.Options.Bias,blocker,true)) continue;
        }
        result=result+Vector(light.Color)*(light.Intensity*weight*cosine/Pi);
    }
    return result;
}
ARVO_LM_HD inline V Integrate(const Scene& job,Surface receiver,Random& random)
{
    V radiance=job.Options.IndirectOnly?V():Direct(job,receiver,random);
    V throughput(1,1,1); Surface current=receiver;
    for(uint32_t bounce=0;bounce<=job.Options.Bounces;++bounce)
    {
        const V direction=Hemisphere(current.Normal,random);
        if(Dot(current.Geometric,direction)<=0) break;
        Surface hit;
        if(!Trace(job,current.Position+current.Geometric*job.Options.Bias,direction,job.Options.Bias*0.1f,1e30f,hit))
        {
            if(bounce>0 || !job.Options.IndirectOnly) radiance=radiance+Mul(throughput,Environment(job,direction));
            break;
        }
        radiance=radiance+Mul(throughput,hit.Emission);
        if(bounce==job.Options.Bounces) break;
        radiance=radiance+Mul(Mul(throughput,hit.Albedo),Direct(job,hit,random));
        throughput=Mul(throughput,hit.Albedo);
        if(Max(throughput)==0) break;
        if(bounce>=2) { const float survive=Clamp(Max(throughput),0.05f,0.95f); if(random.Next()>=survive) break; throughput=throughput/survive; }
        current=hit;
    }
    return radiance;
}
// A triangle clipped against a texel has at most seven vertices.
struct Footprint
{
    V Weights[7]; double Cumulative[5]={};
    double Area=0;
    uint32_t ReceiverIndex=0,Pixel=0,Fans=0;
};
struct Estimate { V Front,Back; uint32_t Accepted=0; };
ARVO_LM_HD inline void SampleRange(const Scene& scene,const Footprint& task,uint32_t first,uint32_t end,Estimate& result)
{
    for(uint32_t sample=first;sample<end;++sample)
    {
        // Independent sample streams allow bounded CUDA launches without changing results.
        Random random{uint64_t(scene.Options.Seed)^((uint64_t(task.ReceiverIndex)+1)*0x9e3779b97f4a7c15ull)^task.Pixel^((uint64_t(sample)+1)*0xd1b54a32d192ed03ull)};
        const double pick=random.Next()*task.Area; uint32_t fan=0;
        while(fan+1<task.Fans && pick>=task.Cumulative[fan]) ++fan;
        const float root=sqrtf(random.Next()),v=random.Next();
        const V w=task.Weights[0]*(1-root)+task.Weights[fan+1]*(root*(1-v))+task.Weights[fan+2]*(root*v);
        const auto& triangle=scene.Receivers[task.ReceiverIndex].Surface;
        const auto surface=Evaluate(scene,triangle,w); if(!surface.Accepted) continue;
        ++result.Accepted; result.Front=result.Front+Integrate(scene,surface,random);
        if(scene.Options.BackFaces) result.Back=result.Back+Integrate(scene,Evaluate(scene,triangle,w,true),random);
    }
}
}
#undef ARVO_LM_HD
