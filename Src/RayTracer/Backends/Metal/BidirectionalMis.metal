// Native Metal complete-path probabilities and MIS. Veach (1997), chapter 10;
// PBRT 3e, "Bidirectional Path Tracing". Independent of every other backend.
// All strategies share one generated density per internal vertexData. Removing a
// common per-vertexData log factor preserves weights and controls numerical range.
constant uint PathZero = 0, PathArea = 1, PathSolidAngle = 2, PathDirac = 3, PathVolume = 4;
constant uint PathSurfaceDestination = 0, PathInfiniteDestination = 1, PathVolumeDestination = 2;
constant uint PathDeltaScattering = 1, PathEndpointNotIntersectable = 2;
constant uint BdptMaximumReflections = 32, BdptMaximumScattering = BdptMaximumReflections + 1;
constant uint BdptMaximumVertices = BdptMaximumScattering + 2;
constant uint BdptMaximumStrategies = BdptMaximumVertices + 1;
constant uint BdptNoMis = 0, BdptBalanceMis = 1, BdptPowerMis = 2;
struct PathDensity
{
    float Log2Value;
    uint Measure;
};
struct BidirectionalVertexPdf
{
    PathDensity Forward, Reverse;
    uint Flags, Reserved0, Reserved1, Reserved2;
};
struct BidirectionalStrategy
{
    PathDensity LightEndpoint, CameraEndpoint;
    float LightSelection, SampleCount;
    uint Enabled, Reserved;
};
struct BidirectionalWeights
{
    float Values[BdptMaximumStrategies];
    uint ActiveStrategies;
};
struct BdptProduct
{
    float Log;
    bool Valid;
};
bool PathDensityValid(PathDensity density)
{
    return density.Measure >= PathArea && density.Measure <= PathVolume && isfinite(density.Log2Value);
}
float PathComparisonLog(PathDensity density)
{
    return density.Measure == PathDirac ? 0 : density.Log2Value;
}
PathDensity MakePathDensity(float value, uint measure)
{
    if (!(value > 0) || !isfinite(value) || measure < PathArea || measure > PathVolume)
        return {};
    return {log2(value), measure};
}
PathDensity ConvertPathDensity(PathDensity p, float3 segment, float3 normal, uint destination)
{
    if (!PathDensityValid(p) || (p.Measure != PathSolidAngle && p.Measure != PathDirac))
        return {};
    if (destination == PathInfiniteDestination)
        return p;
    if (destination != PathSurfaceDestination && destination != PathVolumeDestination)
        return {};
    float largest = max(fabs(segment.x), max(fabs(segment.y), fabs(segment.z)));
    if (!(largest > 0) || !isfinite(largest))
        return {};
    if (p.Measure == PathDirac)
        return p;
    float3 scaled = segment / largest;
    float norm = sqrt(dot(scaled, scaled));
    float cosine = destination == PathVolumeDestination ? 1 : fabs(dot(normal, scaled / norm));
    if (!(cosine > 0) || !isfinite(cosine))
        return {};
    return {p.Log2Value + log2(cosine) - 2 * (log2(largest) + log2(norm)),
            destination == PathVolumeDestination ? PathVolume : PathArea};
}
// For the first surface reached by infinite-source disk emission, project area
// directly. For a camera path escaping to infinity, retain angular measure.
PathDensity ProjectedDiskDensity(PathDensity disk, float3 normal, float3 direction)
{
    if (!PathDensityValid(disk) || disk.Measure != PathArea)
        return {};
    float cosine = fabs(dot(normal, direction));
    if (!(cosine > 0) || !isfinite(cosine))
        return {};
    return {disk.Log2Value + log2(cosine), PathArea};
}
BdptProduct PathProduct(BdptProduct p, PathDensity factor, float reference)
{
    if (!p.Valid || !PathDensityValid(factor))
        return {0, false};
    float value = p.Log + (PathComparisonLog(factor) - reference);
    return {value, isfinite(value)};
}
// Densities refer to the completed path, not stale pre-connection neighbors.
// Dirac factors cancel as one, while zero continuous PDFs remain impossible.
// LightSelection is an ordinary discrete PMF; it always participates in MIS.
BidirectionalWeights BidirectionalMisWeightsLocal(thread const BidirectionalVertexPdf* vertices, uint count,
                                                  thread const BidirectionalStrategy* strategies, uint heuristic)
{
    BidirectionalWeights weights{};
    if (count < 2 || count > BdptMaximumVertices || heuristic > BdptPowerMis)
        return weights;
    float reference[BdptMaximumVertices]{};
    for (uint i = 1; i < count - 1; ++i)
    {
        BidirectionalVertexPdf vertexData = vertices[i];
        bool forward = PathDensityValid(vertexData.Forward), reverse = PathDensityValid(vertexData.Reverse);
        if (forward)
            reference[i] = PathComparisonLog(vertexData.Forward);
        if (reverse)
            reference[i] = forward ? max(reference[i], PathComparisonLog(vertexData.Reverse))
                                   : PathComparisonLog(vertexData.Reverse);
    }
    BdptProduct light[BdptMaximumStrategies], camera[BdptMaximumStrategies];
    for (uint i = 0; i <= count; ++i)
    {
        light[i] = {0, true};
        camera[i] = {0, true};
    }
    for (uint s = 2; s <= count; ++s)
        light[s] = PathProduct(light[s - 1], vertices[s - 1].Forward, reference[s - 1]);
    for (uint s = count - 1; s > 0; --s)
        camera[s - 1] = PathProduct(camera[s], vertices[s - 1].Reverse, reference[s - 1]);
    float scores[BdptMaximumStrategies]{};
    bool active[BdptMaximumStrategies]{};
    float peak = 0;
    for (uint cut = 0; cut <= count; ++cut)
    {
        bool allowed = true;
        if (cut == 0)
            allowed = (vertices[0].Flags & PathEndpointNotIntersectable) == 0;
        else if (cut == count)
            allowed = (vertices[count - 1].Flags & PathEndpointNotIntersectable) == 0;
        else
        {
            if (cut > 1 && (vertices[cut - 1].Flags & PathDeltaScattering) != 0)
                allowed = false;
            if (cut < count - 1 && (vertices[cut].Flags & PathDeltaScattering) != 0)
                allowed = false;
        }
        BidirectionalStrategy candidate = strategies[cut];
        if (!allowed || candidate.Enabled == 0 || !light[cut].Valid || !camera[cut].Valid ||
            !(candidate.LightSelection > 0 && candidate.SampleCount > 0) ||
            !isfinite(candidate.LightSelection) || !isfinite(candidate.SampleCount) ||
            !PathDensityValid(candidate.LightEndpoint) || !PathDensityValid(candidate.CameraEndpoint))
            continue;
        scores[cut] = light[cut].Log + camera[cut].Log + PathComparisonLog(candidate.LightEndpoint) +
                      PathComparisonLog(candidate.CameraEndpoint) + log2(candidate.LightSelection) +
                      log2(candidate.SampleCount);
        if (!isfinite(scores[cut]))
            continue;
        active[cut] = true;
        peak = weights.ActiveStrategies ? max(peak, scores[cut]) : scores[cut];
        ++weights.ActiveStrategies;
    }
    if (!weights.ActiveStrategies)
        return weights;
    float total = 0;
    for (uint cut = 0; cut <= count; ++cut)
        if (active[cut])
        {
            float exponent = heuristic == BdptPowerMis ? 2.0f : 1.0f;
            weights.Values[cut] = heuristic == BdptNoMis ? 1 : exp2(exponent * (scores[cut] - peak));
            total += weights.Values[cut];
        }
    for (uint cut = 0; cut <= count; ++cut)
        weights.Values[cut] /= total;
    return weights;
}

// Diagnostic buffers live in device memory. Production connections assemble
// their completed path in thread memory; both call the same Metal MIS leaf.
BidirectionalWeights BidirectionalMisWeights(device const BidirectionalVertexPdf* vertices, uint count,
                                             device const BidirectionalStrategy* strategies, uint heuristic)
{
    if (count < 2 || count > BdptMaximumVertices || heuristic > BdptPowerMis) return {};
    BidirectionalVertexPdf path[BdptMaximumVertices];
    BidirectionalStrategy cuts[BdptMaximumStrategies];
    for (uint i = 0; i < count; ++i) path[i] = vertices[i];
    for (uint i = 0; i <= count; ++i) cuts[i] = strategies[i];
    return BidirectionalMisWeightsLocal(path, count, cuts, heuristic);
}
