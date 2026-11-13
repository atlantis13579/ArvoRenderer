#include "RayTracer/Backends/CPU/CpuTransport.h"

namespace Arvo::RayTracing::Cpu
{
// Veach (1997), "Robust Monte Carlo Methods for Light Transport Simulation".
// Densities include sample counts: balance is n*p/sum(n*p), power squares n*p.
// A ratio formulation avoids squaring large solid-angle PDFs near an emitter.
float MultipleImportance(float sampledDensity, float competingDensity, MisHeuristic heuristic)
{
    if (heuristic == MisHeuristic::None) return 1.0f;
    if (!(sampledDensity > 0.0f)) return 0.0f;
    if (!(competingDensity > 0.0f)) return 1.0f;
    float ratio = competingDensity / sampledDensity;
    if (heuristic == MisHeuristic::Power) ratio *= ratio;
    return 1.0f / (1.0f + ratio);
}
}
