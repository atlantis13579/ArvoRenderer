#include "Lightmap/Transport.h"
using namespace Arvo::Lightmap::Transport;

extern "C" __global__ void BakeFootprints(Scene scene,const Footprint* tasks,Estimate* estimates,
    unsigned count,unsigned first,unsigned end)
{
    const unsigned index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count) return;
    auto value=estimates[index];
    SampleRange(scene,tasks[index],first,end,value);
    estimates[index]=value;
}
