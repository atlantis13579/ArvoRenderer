#include "Lightmap/Baker.h"
#include "Lightmap/Compute.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <deque>
#include <future>
#include <mutex>
#include <numeric>
#include <thread>

namespace Arvo::Lightmap
{
namespace
{
using namespace ArvoLightmap;
using namespace Transport;
void Require(bool valid,const char* error) { if(!valid) throw std::runtime_error(error); }
template<class T> Span<T> View(const std::vector<T>& values) { return {values.data(),uint32_t(values.size())}; }
class Accelerator
{
    const Job& Source;
    uint32_t Build(uint32_t begin,uint32_t end)
    {
        Node node; Bounds centers;
        for(uint32_t i=begin;i<end;++i) { const auto& t=Source.Triangles[Order[i]]; node.Box.Add(Vector(t.Corners[0].Position)); node.Box.Add(Vector(t.Corners[1].Position)); node.Box.Add(Vector(t.Corners[2].Position)); centers.Add((Vector(t.Corners[0].Position)+Vector(t.Corners[1].Position)+Vector(t.Corners[2].Position))/3); }
        const auto id=uint32_t(Nodes.size()); Nodes.push_back(node);
        if(end-begin<=4) { Nodes[id].Begin=begin; Nodes[id].Count=end-begin; return id; }
        const V span=centers.Hi-centers.Lo; const int axis=span.X>span.Y?(span.X>span.Z?0:2):(span.Y>span.Z?1:2);
        auto key=[&](uint32_t i) { const auto& t=Source.Triangles[i]; const V p=Vector(t.Corners[0].Position)+Vector(t.Corners[1].Position)+Vector(t.Corners[2].Position); return axis==0?p.X:axis==1?p.Y:p.Z; };
        const uint32_t mid=begin+(end-begin)/2;
        std::nth_element(Order.begin()+begin,Order.begin()+mid,Order.begin()+end,[&](auto a,auto b){return key(a)<key(b);});
        const auto left=Build(begin,mid),right=Build(mid,end); Nodes[id].Left=left; Nodes[id].Right=right; return id;
    }
public:
    std::vector<uint32_t> Order;
    std::vector<Node> Nodes;
    explicit Accelerator(const Job& job):Source(job)
    {
        Order.resize(job.Triangles.size()); std::iota(Order.begin(),Order.end(),0u);
        if(!Order.empty()) Build(0,uint32_t(Order.size()));
    }
};
struct Point { double X,Y; V Weights; };
std::vector<Point> Clip(const Receiver& r,uint32_t side,int x,int y)
{
    std::vector<Point> points;
    for(int i=0;i<3;++i) points.push_back({double(r.UV[i].X)*side-x,double(r.UV[i].Y)*side-y,V(i==0?1.f:0.f,i==1?1.f:0.f,i==2?1.f:0.f)});
    for(int plane=0;plane<4 && !points.empty();++plane)
    {
        auto distance=[&](const Point& p) { const double v=plane<2?p.X:p.Y; return plane%2?1-v:v; };
        std::vector<Point> next; auto a=points.back(); double da=distance(a);
        for(const auto& b:points) { const double db=distance(b); if((da>=0)!=(db>=0)) { const double f=da/(da-db); next.push_back({a.X+(b.X-a.X)*f,a.Y+(b.Y-a.Y)*f,a.Weights+(b.Weights-a.Weights)*float(f)}); } if(db>=0) next.push_back(b); a=b; da=db; }
        points=std::move(next);
    }
    return points;
}
void Validate(const Job& job)
{
    const auto& s=job.Options;
    Require(s.Resolution>=4 && s.Resolution<=4096 && s.Samples>=1 && s.Samples<=65536 && s.Bounces<=32 && s.BackFaces<=1 && s.IndirectOnly<=1 && s.Threads<=256 && std::isfinite(s.Bias) && s.Bias>0,"Invalid bake settings");
    Require(Finite(Vector(s.Environment)) && std::min({s.Environment.X,s.Environment.Y,s.Environment.Z})>=0,"Invalid environment");
    Require(job.EnvironmentSide<=4096 && job.EnvironmentFaces.size()==uint64_t(job.EnvironmentSide)*job.EnvironmentSide*6,"Invalid environment cube");
    for(const auto& value:job.EnvironmentFaces) Require(Finite(Vector(value)) && std::min({value.X,value.Y,value.Z})>=0,"Invalid environment texel");
    for(const auto& t:job.Textures) Require(t.Width && t.Height && t.Width<=32768 && t.Height<=32768 && t.Pixels.size()==uint64_t(t.Width)*t.Height*4,"Invalid texture");
    for(const auto& m:job.Materials)
    {
        Require(Finite(RGB(m.Base)) && std::isfinite(m.Base.W) && Finite(Vector(m.Emission)) && std::isfinite(m.AlphaCutoff) && std::isfinite(m.DiffuseScale) && std::isfinite(m.Metallic) && m.DiffuseScale>=0 && m.DiffuseScale<=1,"Invalid material");
        for(auto id:{m.Diffuse,m.Opacity,m.Normal,m.MetallicMap}) Require(id==Missing || id<job.Textures.size(),"Invalid texture index");
    }
    auto triangle=[&](const Triangle& t) { Require(t.Material<job.Materials.size(),"Invalid material index"); for(const auto& v:t.Corners) Require(Finite(Vector(v.Position)) && Max({std::abs(v.Position.X),std::abs(v.Position.Y),std::abs(v.Position.Z)})<1e12f && Finite(Vector(v.Normal)) && Finite(RGB(v.Color)) && std::isfinite(v.Color.W) && std::isfinite(v.UV.X) && std::isfinite(v.UV.Y),"Invalid vertex"); };
    for(const auto& t:job.Triangles) triangle(t);
    for(const auto& l:job.Lights) Require(l.Type<=2 && Finite(Vector(l.Position)) && Finite(Vector(l.Direction)) && (l.Type==1 || Dot(Vector(l.Direction),Vector(l.Direction))>0) && Finite(Vector(l.Color)) && std::isfinite(l.Intensity) && l.Intensity>=0 && std::isfinite(l.Range) && l.Range>=0 && std::isfinite(l.Size.X) && std::isfinite(l.Size.Y) && (l.Type!=2 || (l.Size.X>0 && l.Size.Y>0)),"Invalid light");
    std::vector<uint8_t> occupied(size_t(s.Resolution)*s.Resolution);
    for(const auto& c:job.Charts)
    {
        Require(c.Width && c.Height && c.X<=s.Resolution && c.Y<=s.Resolution && c.Width<=s.Resolution-c.X && c.Height<=s.Resolution-c.Y,"Invalid chart bounds");
        for(uint32_t y=c.Y;y<c.Y+c.Height;++y) for(uint32_t x=c.X;x<c.X+c.Width;++x) { auto& owner=occupied[size_t(y)*s.Resolution+x]; Require(!owner,"Overlapping chart rectangles"); owner=1; }
    }
    for(const auto& r:job.Receivers)
    {
        triangle(r.Surface); Require(r.Chart<job.Charts.size(),"Invalid receiver chart"); const auto& c=job.Charts[r.Chart];
        for(const auto& uv:r.UV) Require(std::isfinite(uv.X) && std::isfinite(uv.Y) && double(uv.X)*s.Resolution>=c.X && double(uv.Y)*s.Resolution>=c.Y && double(uv.X)*s.Resolution<=c.X+c.Width && double(uv.Y)*s.Resolution<=c.Y+c.Height,"Receiver outside chart");
        Require(std::abs((double(r.UV[1].X)-r.UV[0].X)*(double(r.UV[2].Y)-r.UV[0].Y)-(double(r.UV[1].Y)-r.UV[0].Y)*(double(r.UV[2].X)-r.UV[0].X))>0,"Degenerate UV triangle");
    }
}
}
ArvoLightmap::Result Bake(const ArvoLightmap::Job& job,const std::function<void(uint32_t,uint32_t)>& progress,Backend backend)
{
    Validate(job); Accelerator bvh(job);
    std::vector<Image> textures;
    for(const auto& image:job.Textures) textures.push_back({image.Width,image.Height,image.Pixels.data()});
    const Scene scene{job.Options,View(textures),View(job.Materials),View(job.Triangles),job.EnvironmentSide,
        View(job.EnvironmentFaces),View(job.Lights),View(job.Receivers),View(bvh.Nodes),View(bvh.Order)};
    std::unique_ptr<CudaCompute> cuda;
    if(backend==Backend::CUDA) cuda=CreateCudaCompute(scene);
    const auto& settings=job.Options; const auto side=settings.Resolution;
    Result output; output.Resolution=side; output.Front.resize(size_t(side)*side); output.Kinds.resize(output.Front.size());
    if(settings.BackFaces) output.Back.resize(output.Front.size());
    std::vector<double> coverage(output.Front.size()),acceptedCoverage(output.Front.size());
    std::vector<std::vector<uint32_t>> charts(job.Charts.size());
    for(uint32_t i=0;i<job.Receivers.size();++i) charts[job.Receivers[i].Chart].push_back(i);
    std::vector<Footprint> tasks; tasks.reserve(8192);
    auto flush=[&] {
        if(tasks.empty()) return;
        std::vector<Estimate> estimates(tasks.size());
        if(cuda) cuda->Run(tasks,estimates);
        else
        {
            std::atomic<size_t> next{0};
            auto worker=[&] { for(;;) { const auto i=next.fetch_add(1); if(i>=tasks.size()) break; SampleRange(scene,tasks[i],0,settings.Samples,estimates[i]); } };
            const auto count=std::max(1u,std::min(uint32_t(tasks.size()),settings.Threads?settings.Threads:std::max(1u,std::thread::hardware_concurrency())));
            std::vector<std::future<void>> workers;
            for(uint32_t i=1;i<count;++i) workers.push_back(std::async(std::launch::async,worker));
            worker(); for(auto& thread:workers) thread.get();
        }
        for(size_t i=0;i<tasks.size();++i)
        {
            const auto& task=tasks[i]; const auto& value=estimates[i]; const auto at=task.Pixel;
            Require(Finite(value.Front) && Finite(value.Back),"Non-finite lightmap radiance");
            acceptedCoverage[at]+=task.Area*value.Accepted/settings.Samples;
            const float weight=float(task.Area/settings.Samples);
            auto add=[&](Vec4& dst,V v) { dst.X+=v.X*weight; dst.Y+=v.Y*weight; dst.Z+=v.Z*weight; };
            add(output.Front[at],value.Front); if(settings.BackFaces) add(output.Back[at],value.Back);
            if(value.Accepted) output.Kinds[at]=1;
        }
        tasks.clear();
    };
    for(uint32_t ci=0;ci<charts.size();++ci)
    {
        const auto& chart=job.Charts[ci];
        for(auto id:charts[ci])
        {
            const auto& r=job.Receivers[id];
            const int minX=std::max(int(chart.X),int(std::floor(std::min({r.UV[0].X,r.UV[1].X,r.UV[2].X})*side)));
            const int maxX=std::min(int(chart.X+chart.Width),int(std::ceil(std::max({r.UV[0].X,r.UV[1].X,r.UV[2].X})*side)));
            const int minY=std::max(int(chart.Y),int(std::floor(std::min({r.UV[0].Y,r.UV[1].Y,r.UV[2].Y})*side)));
            const int maxY=std::min(int(chart.Y+chart.Height),int(std::ceil(std::max({r.UV[0].Y,r.UV[1].Y,r.UV[2].Y})*side)));
            for(int y=minY;y<maxY;++y) for(int x=minX;x<maxX;++x)
            {
                const auto polygon=Clip(r,side,x,y); if(polygon.size()<3) continue;
                Require(polygon.size()<=7,"Invalid clipped footprint");
                Footprint task; task.ReceiverIndex=id; task.Pixel=uint32_t(y)*side+x; task.Fans=uint32_t(polygon.size()-2);
                for(size_t i=0;i<polygon.size();++i) task.Weights[i]=polygon[i].Weights;
                for(size_t i=1;i+1<polygon.size();++i)
                {
                    const auto& a=polygon[0]; const auto& b=polygon[i]; const auto& c=polygon[i+1];
                    task.Area+=std::abs((b.X-a.X)*(c.Y-a.Y)-(b.Y-a.Y)*(c.X-a.X))*0.5;
                    task.Cumulative[i-1]=task.Area;
                }
                if(task.Area<=1e-15) continue;
                coverage[task.Pixel]+=task.Area;
                Require(coverage[task.Pixel]<=1.001,"Overlapping lightmap UV interiors");
                tasks.push_back(task); if(tasks.size()==8192) flush();
            }
        }
        // Drain periodically so progress describes completed transport work.
        if(ci+1==charts.size() || (ci+1)%256==0) { flush(); if(progress) progress(ci+1,uint32_t(charts.size())); }
    }
    for(const auto& chart:job.Charts)
    {
        std::deque<uint32_t> queue; std::vector<int64_t> source(size_t(chart.Width)*chart.Height,-1);
        for(uint32_t y=0;y<chart.Height;++y) for(uint32_t x=0;x<chart.Width;++x)
        {
            const auto localIndex=y*chart.Width+x,at=(chart.Y+y)*side+chart.X+x;
            if(output.Kinds[at]!=1) continue;
            auto normalize=[&](Vec4& value) { const float w=float(1/acceptedCoverage[at]); value.X*=w;value.Y*=w;value.Z*=w;value.W=1; };
            normalize(output.Front[at]); if(settings.BackFaces) normalize(output.Back[at]);
            source[localIndex]=at; queue.push_back(localIndex);
        }
        // Multi-source dilation is confined to each reserved chart rectangle.
        while(!queue.empty())
        {
            const auto i=queue.front(); queue.pop_front(); const auto x=i%chart.Width,y=i/chart.Width;
            auto fill=[&](uint32_t j) { if(source[j]>=0) return; source[j]=source[i]; queue.push_back(j); const auto at=(chart.Y+j/chart.Width)*side+chart.X+j%chart.Width; output.Front[at]=output.Front[size_t(source[j])]; if(settings.BackFaces) output.Back[at]=output.Back[size_t(source[j])]; output.Kinds[at]=2; };
            if(x) fill(i-1); if(x+1<chart.Width) fill(i+1); if(y) fill(i-chart.Width); if(y+1<chart.Height) fill(i+chart.Width);
        }
    }
    return output;
}
}
