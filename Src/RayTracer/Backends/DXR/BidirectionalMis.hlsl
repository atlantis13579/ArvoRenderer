// DXR's independent complete-path MIS. Veach (1997), chapter 10; PBRT 3e,
// "Bidirectional Path Tracing". No camera/light random walk is reused here.
// Work in log space and remove each internal vertex's common factor. This
// preserves ratios even when direct products would underflow or overflow.
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
struct BdptLogProduct
{
    float Sum;
    bool Positive;
};
PathDensity EmptyPathDensity()
{
    PathDensity p;
    p.Log2Value = 0;
    p.Measure = PathZero;
    return p;
}
bool PathPdfExists(PathDensity p)
{
    return p.Measure >= PathArea && p.Measure <= PathVolume && isfinite(p.Log2Value);
}
float PathComparisonLog(PathDensity p)
{
    return p.Measure == PathDirac ? 0 : p.Log2Value;
}
PathDensity MakePathDensity(float density, uint measure)
{
    PathDensity p = EmptyPathDensity();
    if (density > 0 && isfinite(density) && measure >= PathArea && measure <= PathVolume)
    {
        p.Log2Value = log2(density);
        p.Measure = measure;
    }
    return p;
}
// Convert angular density to destination geometric area; never use shading
// normals. Infinite destinations and Dirac constraints retain their measures.
PathDensity ConvertPathDensity(PathDensity pdf, float3 segment, float3 geometricNormal, uint destination)
{
    if (!PathPdfExists(pdf) || (pdf.Measure != PathSolidAngle && pdf.Measure != PathDirac))
        return EmptyPathDensity();
    if (destination == PathInfiniteDestination)
        return pdf;
    if (destination != PathSurfaceDestination && destination != PathVolumeDestination)
        return EmptyPathDensity();
    float largest = max(abs(segment.x), max(abs(segment.y), abs(segment.z)));
    if (!(largest > 0) || !isfinite(largest))
        return EmptyPathDensity();
    if (pdf.Measure == PathDirac)
        return pdf;
    float3 local = segment / largest;
    float norm = sqrt(dot(local, local));
    float geometricCosine = destination == PathVolumeDestination ? 1 :
        abs(dot(geometricNormal, local / norm));
    if (!(geometricCosine > 0) || !isfinite(geometricCosine))
        return EmptyPathDensity();
    pdf.Log2Value = pdf.Log2Value + log2(geometricCosine) - 2 * (log2(largest) + log2(norm));
    pdf.Measure = destination == PathVolumeDestination ? PathVolume : PathArea;
    return pdf;
}
PathDensity ProjectedDiskDensity(PathDensity pdf, float3 normal, float3 unitRay)
{
    if (!PathPdfExists(pdf) || pdf.Measure != PathArea)
        return EmptyPathDensity();
    float projection = abs(dot(normal, unitRay));
    if (!(projection > 0) || !isfinite(projection))
        return EmptyPathDensity();
    pdf.Log2Value += log2(projection);
    return pdf;
}
BdptLogProduct ExtendPathProduct(BdptLogProduct value, PathDensity density, float baseline)
{
    value.Positive = value.Positive && PathPdfExists(density);
    if (value.Positive)
        value.Sum += PathComparisonLog(density) - baseline;
    value.Positive = value.Positive && isfinite(value.Sum);
    return value;
}
// Forward[i] samples x[i] from x[i-1], Reverse[i] from x[i+1]. All records must
// be evaluated on the completed path, including the changed connection BSDFs.
// Dirac factors cancel to one for comparisons; true zero factors stay absent.
BidirectionalWeights BidirectionalMisWeightsLocal(BidirectionalVertexPdf vertices[BdptMaximumVertices],
                                                  uint count,
                                                  BidirectionalStrategy strategies[BdptMaximumStrategies],
                                                  uint heuristic)
{
    BidirectionalWeights result;
    result.ActiveStrategies = 0;
    for (uint i = 0; i < BdptMaximumStrategies; ++i)
        result.Values[i] = 0;
    if (count < 2 || count > BdptMaximumVertices || heuristic > BdptPowerMis)
        return result;
    float common[BdptMaximumVertices];
    for (uint i = 0; i < count; ++i)
        common[i] = 0;
    for (uint i = 1; i < count - 1; ++i)
    {
        BidirectionalVertexPdf vertex = vertices[i];
        bool forward = PathPdfExists(vertex.Forward), reverse = PathPdfExists(vertex.Reverse);
        if (forward)
            common[i] = PathComparisonLog(vertex.Forward);
        if (reverse)
            common[i] = forward ? max(common[i], PathComparisonLog(vertex.Reverse))
                                : PathComparisonLog(vertex.Reverse);
    }
    BdptLogProduct fromLight[BdptMaximumStrategies], fromCamera[BdptMaximumStrategies];
    for (uint i = 0; i <= count; ++i)
    {
        fromLight[i].Sum = 0;
        fromLight[i].Positive = true;
        fromCamera[i] = fromLight[i];
    }
    for (uint cut = 2; cut <= count; ++cut)
        fromLight[cut] =
            ExtendPathProduct(fromLight[cut - 1], vertices[cut - 1].Forward, common[cut - 1]);
    for (uint end = count - 1; end > 0; --end)
        fromCamera[end - 1] =
            ExtendPathProduct(fromCamera[end], vertices[end - 1].Reverse, common[end - 1]);
    float logScores[BdptMaximumStrategies];
    bool included[BdptMaximumStrategies];
    float largest = 0;
    for (uint cut = 0; cut <= count; ++cut)
    {
        included[cut] = false;
        logScores[cut] = 0;
        bool permitted = true;
        if (cut == 0)
            permitted = (vertices[0].Flags & PathEndpointNotIntersectable) == 0;
        else if (cut == count)
            permitted = (vertices[count - 1].Flags & PathEndpointNotIntersectable) == 0;
        else
        {
            if (cut > 1 && (vertices[cut - 1].Flags & PathDeltaScattering) != 0)
                permitted = false;
            if (cut < count - 1 && (vertices[cut].Flags & PathDeltaScattering) != 0)
                permitted = false;
        }
        BidirectionalStrategy candidate = strategies[cut];
        if (!permitted || candidate.Enabled == 0 || !fromLight[cut].Positive || !fromCamera[cut].Positive ||
            !(candidate.LightSelection > 0 && candidate.SampleCount > 0) ||
            !isfinite(candidate.LightSelection) || !isfinite(candidate.SampleCount) ||
            !PathPdfExists(candidate.LightEndpoint) || !PathPdfExists(candidate.CameraEndpoint))
            continue;
        logScores[cut] = fromLight[cut].Sum + fromCamera[cut].Sum +
                         PathComparisonLog(candidate.LightEndpoint) +
                         PathComparisonLog(candidate.CameraEndpoint) + log2(candidate.LightSelection) +
                         log2(candidate.SampleCount);
        if (!isfinite(logScores[cut]))
            continue;
        included[cut] = true;
        largest = result.ActiveStrategies ? max(largest, logScores[cut]) : logScores[cut];
        ++result.ActiveStrategies;
    }
    if (!result.ActiveStrategies)
        return result;
    float denominator = 0;
    for (uint cut = 0; cut <= count; ++cut)
        if (included[cut])
        {
            float power = heuristic == BdptPowerMis ? 2.0f : 1.0f;
            result.Values[cut] = heuristic == BdptNoMis ? 1 : exp2((logScores[cut] - largest) * power);
            denominator += result.Values[cut];
        }
    for (uint cut = 0; cut <= count; ++cut)
        result.Values[cut] /= denominator;
    return result;
}

// Diagnostic-buffer adapter. The production integrator assembles the complete
// path in local arrays, retaining this same numerical implementation.
BidirectionalWeights BidirectionalMisWeights(StructuredBuffer<BidirectionalVertexPdf> vertices,
                                             uint firstVertex, uint count,
                                             StructuredBuffer<BidirectionalStrategy> strategies,
                                             uint firstStrategy, uint heuristic)
{
    BidirectionalWeights empty;
    empty.ActiveStrategies = 0;
    for (uint k = 0; k < BdptMaximumStrategies; ++k) empty.Values[k] = 0;
    if (count < 2 || count > BdptMaximumVertices || heuristic > BdptPowerMis) return empty;
    BidirectionalVertexPdf path[BdptMaximumVertices];
    BidirectionalStrategy cuts[BdptMaximumStrategies];
    for (uint i = 0; i < count; ++i) path[i] = vertices[firstVertex + i];
    for (uint j = 0; j <= count; ++j) cuts[j] = strategies[firstStrategy + j];
    return BidirectionalMisWeightsLocal(path, count, cuts, heuristic);
}
