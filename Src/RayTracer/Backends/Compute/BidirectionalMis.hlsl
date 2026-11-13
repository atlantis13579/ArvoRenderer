// Independent Compute path densities and BDPT strategy MIS. Veach (1997), ch.10;
// PBRT 3e, "Bidirectional Path Tracing". A cut s generates vertices [0,s) from
// light and [s,n) from camera. Dirac factors cancel as unit comparison factors;
// actual zero PDFs stay zero and delta-adjacent deterministic cuts are excluded.
static const uint PathZero = 0, PathArea = 1, PathSolidAngle = 2, PathDirac = 3, PathVolume = 4;
static const uint PathSurfaceDestination = 0, PathInfiniteDestination = 1, PathVolumeDestination = 2;
static const uint PathDeltaScattering = 1, PathEndpointNotIntersectable = 2;
static const uint BdptMaximumReflections = 32, BdptMaximumScattering = BdptMaximumReflections + 1;
static const uint BdptMaximumVertices = BdptMaximumScattering + 2;
static const uint BdptMaximumStrategies = BdptMaximumVertices + 1;
static const uint BdptNoMis = 0, BdptBalanceMis = 1, BdptPowerMis = 2;
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
PathDensity ZeroPathDensity()
{
    PathDensity p;
    p.Log2Value = 0;
    p.Measure = PathZero;
    return p;
}
bool ValidPathDensity(PathDensity p)
{
    return p.Measure >= PathArea && p.Measure <= PathVolume && isfinite(p.Log2Value);
}
float BdptComparisonLog(PathDensity p)
{
    return p.Measure == PathDirac ? 0 : p.Log2Value;
}
PathDensity MakePathDensity(float value, uint measure)
{
    PathDensity result = ZeroPathDensity();
    if (!(value > 0) || !isfinite(value) || measure < PathArea || measure > PathVolume)
        return result;
    result.Log2Value = log2(value);
    result.Measure = measure;
    return result;
}
// Geometric normal and direction are unit. Infinite endpoints retain angular
// measure. Scale before normalizing the displacement; work in logs to avoid
// losing tiny positive area PDFs or overflowing a squared world distance.
PathDensity ConvertPathDensity(PathDensity p, float3 displacement, float3 normal, uint destination)
{
    if (!ValidPathDensity(p) || (p.Measure != PathSolidAngle && p.Measure != PathDirac))
        return ZeroPathDensity();
    if (destination == PathInfiniteDestination)
        return p;
    if (destination != PathSurfaceDestination && destination != PathVolumeDestination)
        return ZeroPathDensity();
    float scale = max(abs(displacement.x), max(abs(displacement.y), abs(displacement.z)));
    if (!(scale > 0) || !isfinite(scale))
        return ZeroPathDensity();
    if (p.Measure == PathDirac)
        return p;
    float3 scaled = displacement / scale;
    float magnitude = sqrt(dot(scaled, scaled));
    float cosine = destination == PathVolumeDestination ? 1 : abs(dot(normal, scaled / magnitude));
    if (!(cosine > 0) || !isfinite(cosine))
        return ZeroPathDensity();
    p.Log2Value += log2(cosine) - 2 * (log2(scale) + log2(magnitude));
    p.Measure = destination == PathVolumeDestination ? PathVolume : PathArea;
    return p;
}
PathDensity ProjectedDiskDensity(PathDensity disk, float3 normal, float3 direction)
{
    if (!ValidPathDensity(disk) || disk.Measure != PathArea)
        return ZeroPathDensity();
    float cosine = abs(dot(normal, direction));
    if (!(cosine > 0) || !isfinite(cosine))
        return ZeroPathDensity();
    disk.Log2Value += log2(cosine);
    return disk;
}
BdptProduct BdptAppend(BdptProduct product, PathDensity factor, float common)
{
    product.Valid = product.Valid && ValidPathDensity(factor);
    if (product.Valid)
        product.Log += BdptComparisonLog(factor) - common;
    product.Valid = product.Valid && isfinite(product.Log);
    return product;
}
// All factors describe one completed path. Re-evaluate connection-neighbor
// BSDF PDFs before calling. Endpoint factors allow NEE/camera-specific sampling.
// LightSelection is a real PMF and is retained even though it is discrete.
BidirectionalWeights BidirectionalMisWeightsLocal(BidirectionalVertexPdf vertices[BdptMaximumVertices],
                                                   uint count,
                                                   BidirectionalStrategy strategies[BdptMaximumStrategies],
                                                   uint heuristic)
{
    BidirectionalWeights weights;
    weights.ActiveStrategies = 0;
    for (uint i = 0; i < BdptMaximumStrategies; ++i)
        weights.Values[i] = 0;
    if (count < 2 || count > BdptMaximumVertices || heuristic > BdptPowerMis)
        return weights;
    float reference[BdptMaximumVertices];
    for (uint i = 0; i < count; ++i)
        reference[i] = 0;
    for (uint i = 1; i + 1 < count; ++i)
    {
        BidirectionalVertexPdf vertex = vertices[i];
        bool forward = ValidPathDensity(vertex.Forward), reverse = ValidPathDensity(vertex.Reverse);
        reference[i] =
            forward ? BdptComparisonLog(vertex.Forward) : (reverse ? BdptComparisonLog(vertex.Reverse) : 0);
        if (forward && reverse)
            reference[i] = max(reference[i], BdptComparisonLog(vertex.Reverse));
    }
    BdptProduct prefix[BdptMaximumStrategies], suffix[BdptMaximumStrategies];
    for (uint i = 0; i <= count; ++i)
    {
        prefix[i].Log = 0;
        prefix[i].Valid = true;
        suffix[i] = prefix[i];
    }
    for (uint s = 2; s <= count; ++s)
        prefix[s] = BdptAppend(prefix[s - 1], vertices[s - 1].Forward, reference[s - 1]);
    for (uint s = count - 1; s > 0; --s)
        suffix[s - 1] = BdptAppend(suffix[s], vertices[s - 1].Reverse, reference[s - 1]);
    float scores[BdptMaximumStrategies];
    bool active[BdptMaximumStrategies];
    float maximum = 0;
    for (uint s = 0; s <= count; ++s)
    {
        active[s] = false;
        scores[s] = 0;
        bool allowed = true;
        if (s == 0)
            allowed = (vertices[0].Flags & PathEndpointNotIntersectable) == 0;
        else if (s == count)
            allowed = (vertices[count - 1].Flags & PathEndpointNotIntersectable) == 0;
        else
        {
            if (s > 1 && (vertices[s - 1].Flags & PathDeltaScattering) != 0)
                allowed = false;
            if (s + 1 < count && (vertices[s].Flags & PathDeltaScattering) != 0)
                allowed = false;
        }
        BidirectionalStrategy strategy = strategies[s];
        if (!allowed || strategy.Enabled == 0 || !(strategy.LightSelection > 0 && strategy.SampleCount > 0) ||
            !isfinite(strategy.LightSelection) || !isfinite(strategy.SampleCount) || !prefix[s].Valid ||
            !suffix[s].Valid || !ValidPathDensity(strategy.LightEndpoint) ||
            !ValidPathDensity(strategy.CameraEndpoint))
            continue;
        scores[s] = prefix[s].Log + suffix[s].Log + BdptComparisonLog(strategy.LightEndpoint) +
                    BdptComparisonLog(strategy.CameraEndpoint) + log2(strategy.LightSelection) +
                    log2(strategy.SampleCount);
        if (!isfinite(scores[s]))
            continue;
        active[s] = true;
        maximum = weights.ActiveStrategies > 0 ? max(maximum, scores[s]) : scores[s];
        ++weights.ActiveStrategies;
    }
    if (weights.ActiveStrategies == 0)
        return weights;
    float sum = 0;
    for (uint s = 0; s <= count; ++s)
        if (active[s])
        {
            float exponent = heuristic == BdptPowerMis ? 2.0f : 1.0f;
            weights.Values[s] = heuristic == BdptNoMis ? 1 : exp2(exponent * (scores[s] - maximum));
            sum += weights.Values[s];
        }
    for (uint s = 0; s <= count; ++s)
        weights.Values[s] /= sum;
    return weights;
}

// Buffer adapter for the diagnostic probe. Production BDPT builds the same
// completed-path factors in thread-local arrays, without a scratch UAV roundtrip.
BidirectionalWeights BidirectionalMisWeights(StructuredBuffer<BidirectionalVertexPdf> vertices,
                                             uint vertexOffset, uint count,
                                             StructuredBuffer<BidirectionalStrategy> strategies,
                                             uint strategyOffset, uint heuristic)
{
    if (count < 2 || count > BdptMaximumVertices || heuristic > BdptPowerMis)
    {
        BidirectionalWeights empty;
        empty.ActiveStrategies = 0;
        for (uint k = 0; k < BdptMaximumStrategies; ++k) empty.Values[k] = 0;
        return empty;
    }
    BidirectionalVertexPdf localVertices[BdptMaximumVertices];
    BidirectionalStrategy localStrategies[BdptMaximumStrategies];
    for (uint i = 0; i < count && i < BdptMaximumVertices; ++i)
        localVertices[i] = vertices[vertexOffset + i];
    for (uint j = 0; j <= count && j < BdptMaximumStrategies; ++j)
        localStrategies[j] = strategies[strategyOffset + j];
    return BidirectionalMisWeightsLocal(localVertices, count, localStrategies, heuristic);
}
