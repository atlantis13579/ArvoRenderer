#pragma once
#include "Lightmap/Transport.h"
#include <memory>

namespace Arvo::Lightmap
{
class CudaCompute
{
public:
    virtual ~CudaCompute()=default;
    virtual void Run(const std::vector<Transport::Footprint>&,std::vector<Transport::Estimate>&)=0;
};
// A private CUDA compute context, with no presentation or graphics interoperability.
std::unique_ptr<CudaCompute> CreateCudaCompute(const Transport::Scene&);
}
