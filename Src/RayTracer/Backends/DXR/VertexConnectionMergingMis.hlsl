// DXR VCM: an independent complete-path MIS evaluator for offline rendering.
// Interactive preview does not construct connected or merged paths.
// Georgiev et al., "Light Transport Simulation with Vertex Connection and
// Merging" (2012); "Implementing Vertex Connection and Merging", rev. 2:
// https://www.iliyan.com/publications/ImplementingVCM/ImplementingVCM_TechRep2012_rev2.pdf
//
// Vertices run from light x[0] to camera x[n-1]. Connection s samples
// Forward[1..s-1] * Reverse[s..n-2], plus its endpoint factors. Merging s
// samples x[s-1] twice, adding the PHYSICAL Reverse[s-1] and pi*r^2.
// SampleCount is the attempted population, including unsuccessful walks.
// Delta events exclude strategies; their branch PMFs and Jacobians remain.
static const uint VcmMaximumReflections = 32;
static const uint VcmMaximumScattering = VcmMaximumReflections + 1;
static const uint VcmMaximumVertices = VcmMaximumScattering + 2;
static const uint VcmMaximumStrategies = VcmMaximumVertices + 1;
static const uint VcmDeltaScattering = 1, VcmEndpointNotIntersectable = 2, VcmVolumeScattering = 4;
static const uint VcmConnections = 1, VcmMergings = 2;
static const uint VcmEqualMis = 0, VcmBalanceMis = 1, VcmPowerMis = 2;
static const double VcmPi = 3.141592653589793238462643383279502884L;
static const double VcmLn2 = 0.693147180559945309417232121458176568L;
static const double VcmInverseLn2 = 1.442695040888963407359924681001892137L;

struct VcmDensity
{
    double Log2Value;
    uint Positive, Reserved;
};
struct VcmVertexPdf
{
    VcmDensity Forward, Reverse;
    uint Flags, Reserved0, Reserved1, Reserved2;
};
struct VcmStrategy
{
    VcmDensity LightEndpoint, CameraEndpoint;
    double LightSelection, SampleCount;
    uint Enabled, Reserved0, Reserved1, Reserved2;
};
struct VcmWeights
{
    double Connection[VcmMaximumStrategies], Merging[VcmMaximumStrategies];
    uint ActiveConnections, ActiveMergings, Reserved0, Reserved1;
};

// HLSL transcendental intrinsics narrow doubles. Decode IEEE-754 exponents and
// use short convergent series instead. Compile with -Gis so common density
// factors cancel BEFORE summation. L-suffixed literals keep binary64 precision.
// https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-appendix-grammar
static const uint VcmDoubleExponentMask = 0x7ff00000u, VcmDoubleFractionMask = 0x000fffffu;
static const uint VcmDoubleExponentShift = 20, VcmDoubleExponentBias = 1023;
static const int VcmDoubleMinimumExponent = -1022, VcmDoubleFractionBits = 52;
static const uint VcmFloatExponentMask = 0xffu, VcmFloatFractionMask = 0x007fffffu;
static const uint VcmFloatFractionBits = 23, VcmFloatSignMask = 0x80000000u;
static const int VcmFloatExponentBias = 127;

// Keep iterative density evaluation as control flow. Fully expanding every
// series/tree traversal at every connection duplicates substantial binary64
// code in the ten-stage ray-tracing library. Iteration order and budgets stay
// unchanged.
// https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-for

bool VcmFinite(double value)
{
    uint low, high;
    asuint(value, low, high);
    return (high & VcmDoubleExponentMask) != VcmDoubleExponentMask;
}
double VcmPowerOfTwo(int exponent)
{
    return asdouble(0, uint(exponent + int(VcmDoubleExponentBias)) << VcmDoubleExponentShift);
}
double VcmWidenFloat(float value)
{
    // Preserve even float subnormals under native shader flush-to-zero rules.
    uint bits = asuint(value);
    uint exponent = (bits >> VcmFloatFractionBits) & VcmFloatExponentMask;
    uint fraction = bits & VcmFloatFractionMask;
    if (exponent == VcmFloatExponentMask)
        return asdouble(0, VcmDoubleExponentMask | (fraction != 0 ? 1u : 0u));
    double magnitude;
    if (exponent == 0)
        magnitude = double(fraction) * VcmPowerOfTwo(1 - VcmFloatExponentBias - int(VcmFloatFractionBits));
    else
        magnitude = double(fraction | (1u << VcmFloatFractionBits)) *
                    VcmPowerOfTwo(int(exponent) - VcmFloatExponentBias - int(VcmFloatFractionBits));
    return (bits & VcmFloatSignMask) != 0 ? -magnitude : magnitude;
}
double VcmLog2Positive(double value)
{
    uint low, high;
    asuint(value, low, high);
    int exponentAdjustment = 0;
    if ((high & VcmDoubleExponentMask) == 0)
    {
        value *= VcmPowerOfTwo(VcmDoubleFractionBits);
        exponentAdjustment = -VcmDoubleFractionBits;
        asuint(value, low, high);
    }
    int exponent = int((high & VcmDoubleExponentMask) >> VcmDoubleExponentShift) -
                   int(VcmDoubleExponentBias) + exponentAdjustment;
    double mantissa =
        asdouble(low, (high & VcmDoubleFractionMask) | (VcmDoubleExponentBias << VcmDoubleExponentShift));
    if (mantissa > 1.5L)
    {
        mantissa *= 0.5L;
        ++exponent;
    }
    double ratio = (mantissa - 1) / (mantissa + 1), square = ratio * ratio;
    double power = ratio, series = ratio;
    // Centering also preserves logs immediately below one. |ratio| <= 1/5;
    // sixteen odd terms leave truncation error below binary64 epsilon.
    static const uint LogSeriesTerms = 16;
    [loop]
    for (uint term = 1; term < LogSeriesTerms; ++term)
    {
        power *= square;
        series += power / double(2 * term + 1);
    }
    return double(exponent) + (2 * VcmInverseLn2) * series;
}
double VcmRelativeExp2(double logarithm)
{
    static const int RoundingZeroExponent = VcmDoubleMinimumExponent - VcmDoubleFractionBits - 1;
    static const int SubnormalScaleBits = 64, ExpSeriesDegree = 18;
    if (logarithm <= double(RoundingZeroExponent))
        return 0;
    if (logarithm >= 0)
        return 1;
    int exponent = int(logarithm);
    double remainder = logarithm - double(exponent);
    if (remainder < -0.5L)
    {
        --exponent;
        remainder += 1;
    }
    double argument = remainder * VcmLn2, polynomial = 1;
    [loop]
    for (int degree = ExpSeriesDegree; degree > 0; --degree)
        polynomial = 1 + argument * polynomial / double(degree);
    if (exponent < VcmDoubleMinimumExponent)
    {
        // Keep the intermediate normal, then round to a subnormal exactly once.
        double normal = polynomial * VcmPowerOfTwo(exponent + SubnormalScaleBits);
        return normal * VcmPowerOfTwo(-SubnormalScaleBits);
    }
    return polynomial * VcmPowerOfTwo(exponent);
}
VcmDensity VcmLogDensity(double logarithm, uint positive)
{
    VcmDensity result;
    result.Positive = positive != 0 && VcmFinite(logarithm) ? 1 : 0;
    result.Log2Value = result.Positive != 0 ? logarithm : 0;
    result.Reserved = 0;
    return result;
}
bool VcmHasDensity(VcmDensity density)
{
    return density.Positive != 0 && VcmFinite(density.Log2Value);
}
VcmDensity VcmDensityFromValue(double value)
{
    if (value > 0 && VcmFinite(value))
        return VcmLogDensity(VcmLog2Positive(value), 1);
    return VcmLogDensity(0, 0);
}
VcmDensity VcmSurfaceArrival(VcmDensity directional, float3 displacement, float3 geometricNormal)
{
    double x = VcmWidenFloat(displacement.x), y = VcmWidenFloat(displacement.y),
           z = VcmWidenFloat(displacement.z);
    double squaredDistance = x * x + y * y + z * z;
    double projection = x * VcmWidenFloat(geometricNormal.x) + y * VcmWidenFloat(geometricNormal.y) +
                        z * VcmWidenFloat(geometricNormal.z);
    projection = projection < 0 ? -projection : projection;
    if (!VcmHasDensity(directional) || !(squaredDistance > 0) || !VcmFinite(squaredDistance) ||
        !(projection > 0) || !VcmFinite(projection))
        return VcmLogDensity(0, 0);
    // |n.d| / |d|^3 is the angular-to-area Jacobian, without a square root.
    static const double SquaredDistanceLogPower = 1.5L;
    return VcmLogDensity(directional.Log2Value + VcmLog2Positive(projection) -
                             SquaredDistanceLogPower * VcmLog2Positive(squaredDistance),
                         1);
}
VcmDensity VcmDiskArrival(VcmDensity projectedDisk, float3 geometricNormal, float3 unitDirection)
{
    double cosine = VcmWidenFloat(geometricNormal.x) * VcmWidenFloat(unitDirection.x) +
                    VcmWidenFloat(geometricNormal.y) * VcmWidenFloat(unitDirection.y) +
                    VcmWidenFloat(geometricNormal.z) * VcmWidenFloat(unitDirection.z);
    cosine = cosine < 0 ? -cosine : cosine;
    if (!VcmHasDensity(projectedDisk) || !(cosine > 0) || !VcmFinite(cosine))
        return VcmLogDensity(0, 0);
    return VcmLogDensity(projectedDisk.Log2Value + VcmLog2Positive(cosine), 1);
}
VcmDensity VcmDiskArea(double radius)
{
    if (radius > 0 && VcmFinite(radius))
        return VcmLogDensity(VcmLog2Positive(VcmPi) + 2 * VcmLog2Positive(radius), 1);
    return VcmLogDensity(0, 0);
}
VcmDensity VcmSphereVolume(double radius)
{
    if (radius > 0 && VcmFinite(radius))
        return VcmLogDensity(VcmLog2Positive(4 * VcmPi / 3) + 3 * VcmLog2Positive(radius), 1);
    return VcmLogDensity(0, 0);
}

// Move a cut by replacing one leaf in a balanced product tree. Rebuild its
// ancestors instead of subtracting the old total: a huge old log must not erase
// small factors when it leaves the product. Missing factors have their own count
// and can become valid again after the cut moves past a zero-probability edge.
static const uint VcmProductLeaves = 64; // Smallest power of two covering 35 vertices.
static const uint VcmProductNodes = 2 * VcmProductLeaves;
struct VcmProductTree
{
    double Logarithm[VcmProductNodes];
    uint Missing[VcmProductNodes];
};
struct VcmScore
{
    double Logarithm;
    uint Valid;
};
double VcmCommonFactor(VcmVertexPdf vertex)
{
    bool forward = VcmHasDensity(vertex.Forward), reverse = VcmHasDensity(vertex.Reverse);
    if (forward && reverse)
        return vertex.Forward.Log2Value > vertex.Reverse.Log2Value ? vertex.Forward.Log2Value
                                                                   : vertex.Reverse.Log2Value;
    return forward ? vertex.Forward.Log2Value : reverse ? vertex.Reverse.Log2Value : 0;
}
void VcmReplaceLeaf(inout VcmProductTree tree, uint vertex, VcmDensity density, double common)
{
    uint node = VcmProductLeaves + vertex;
    bool present = VcmHasDensity(density);
    precise double residual = present ? density.Log2Value - common : 0;
    tree.Logarithm[node] = residual;
    tree.Missing[node] = present ? 0 : 1;
    [loop]
    while (node > 1)
    {
        node /= 2;
        tree.Logarithm[node] = tree.Logarithm[2 * node] + tree.Logarithm[2 * node + 1];
        tree.Missing[node] = tree.Missing[2 * node] + tree.Missing[2 * node + 1];
    }
}
VcmScore VcmAppendScore(VcmScore score, VcmDensity density)
{
    score.Valid = score.Valid != 0 && VcmHasDensity(density) ? 1 : 0;
    if (score.Valid != 0)
        score.Logarithm += density.Log2Value;
    score.Valid = score.Valid != 0 && VcmFinite(score.Logarithm) ? 1 : 0;
    return score;
}
VcmScore VcmEndpointScore(VcmScore path, VcmStrategy strategy)
{
    if (strategy.Enabled == 0 || !(strategy.LightSelection > 0 && strategy.LightSelection <= 1) ||
        !(strategy.SampleCount > 0) || !VcmFinite(strategy.SampleCount))
        path.Valid = 0;
    path = VcmAppendScore(path, strategy.LightEndpoint);
    path = VcmAppendScore(path, strategy.CameraEndpoint);
    path = VcmAppendScore(path, VcmDensityFromValue(strategy.LightSelection));
    return VcmAppendScore(path, VcmDensityFromValue(strategy.SampleCount));
}
bool VcmConnectionEligible(VcmVertexPdf vertices[VcmMaximumVertices], uint count, uint cut)
{
    if (cut == 0)
        return (vertices[0].Flags & VcmEndpointNotIntersectable) == 0;
    if (cut == count)
        return (vertices[count - 1].Flags & VcmEndpointNotIntersectable) == 0;
    if (cut > 1 && (vertices[cut - 1].Flags & VcmDeltaScattering) != 0)
        return false;
    return cut + 1 >= count || (vertices[cut].Flags & VcmDeltaScattering) == 0;
}
struct VcmMisEvaluation
{
    VcmWeights Weights;
    VcmDensity SelectedLogWeight;
};
VcmMisEvaluation VcmPackMis(VcmWeights weights, VcmDensity selected)
{
    VcmMisEvaluation evaluation;
    evaluation.Weights = weights;
    evaluation.SelectedLogWeight = selected;
    return evaluation;
}
VcmMisEvaluation VcmEvaluateMis(VcmVertexPdf vertices[VcmMaximumVertices], uint count,
                                VcmStrategy connections[VcmMaximumStrategies],
                                VcmStrategy mergings[VcmMaximumStrategies], double radius, uint heuristic,
                                uint families, uint selectedCut, uint selectedMerge)
{
    VcmWeights result;
    VcmDensity selected = VcmLogDensity(0, 0);
    result.ActiveConnections = result.ActiveMergings = result.Reserved0 = result.Reserved1 = 0;
    [loop]
    for (uint slot = 0; slot < VcmMaximumStrategies; ++slot)
        result.Connection[slot] = result.Merging[slot] = 0;
    if (count < 2 || count > VcmMaximumVertices || heuristic > VcmPowerMis ||
        (families & ~(VcmConnections | VcmMergings)) != 0 || !(radius >= 0) || !VcmFinite(radius))
        return VcmPackMis(result, selected);

    VcmProductTree tree;
    [loop]
    for (uint node = 0; node < VcmProductNodes; ++node)
    {
        tree.Logarithm[node] = 0;
        tree.Missing[node] = 0;
    }
    [loop]
    for (uint vertex = 0; vertex + 1 < count; ++vertex)
    {
        double common = vertex == 0 ? 0 : VcmCommonFactor(vertices[vertex]);
        VcmReplaceLeaf(tree, vertex, vertices[vertex].Reverse, common);
    }
    VcmScore connection[VcmMaximumStrategies], merging[VcmMaximumStrategies];
    VcmDensity area = VcmDiskArea(radius), volume = VcmSphereVolume(radius);
    double largest = 0;
    uint active = 0;
    [loop]
    for (uint cut = 0; cut <= count; ++cut)
    {
        if (cut > 0)
        {
            uint vertex = cut - 1;
            VcmDensity replacement = vertices[vertex].Forward;
            if (vertex == 0)
                replacement = VcmLogDensity(0, 1);
            double common = vertex > 0 && vertex + 1 < count ? VcmCommonFactor(vertices[vertex]) : 0;
            VcmReplaceLeaf(tree, vertex, replacement, common);
        }
        VcmScore path;
        path.Logarithm = tree.Logarithm[1];
        path.Valid = tree.Missing[1] == 0 && VcmFinite(path.Logarithm) ? 1 : 0;
        connection[cut].Logarithm = merging[cut].Logarithm = 0;
        connection[cut].Valid = merging[cut].Valid = 0;
        if ((families & VcmConnections) != 0 && VcmConnectionEligible(vertices, count, cut))
            connection[cut] = VcmEndpointScore(path, connections[cut]);
        if ((families & VcmMergings) != 0 && cut >= 2 && cut < count &&
            (vertices[cut - 1].Flags & VcmDeltaScattering) == 0)
        {
            VcmScore overlap = VcmAppendScore(path, vertices[cut - 1].Reverse);
            VcmDensity measure = area;
            if ((vertices[cut - 1].Flags & VcmVolumeScattering) != 0)
                measure = volume;
            merging[cut] = VcmEndpointScore(VcmAppendScore(overlap, measure), mergings[cut]);
        }
        if (connection[cut].Valid != 0)
        {
            if (active == 0 || connection[cut].Logarithm > largest)
                largest = connection[cut].Logarithm;
            ++active;
            ++result.ActiveConnections;
        }
        if (merging[cut].Valid != 0)
        {
            if (active == 0 || merging[cut].Logarithm > largest)
                largest = merging[cut].Logarithm;
            ++active;
            ++result.ActiveMergings;
        }
    }
    if (active == 0)
        return VcmPackMis(result, selected);
    double sum = 0, power = heuristic == VcmPowerMis ? 2 : 1;
    [loop]
    for (uint cut = 0; cut <= count; ++cut)
    {
        if (connection[cut].Valid != 0)
            result.Connection[cut] =
                heuristic == VcmEqualMis ? 1 : VcmRelativeExp2(power * (connection[cut].Logarithm - largest));
        if (merging[cut].Valid != 0)
            result.Merging[cut] =
                heuristic == VcmEqualMis ? 1 : VcmRelativeExp2(power * (merging[cut].Logarithm - largest));
        sum += result.Connection[cut] + result.Merging[cut];
    }
    // Keep the selected weight in log space until the contribution's kernel
    // normalization is applied. A tiny VM weight can cancel a tiny disk area.
    if (selectedCut <= count)
    {
        VcmScore score = connection[selectedCut];
        if (selectedMerge != 0)
            score = merging[selectedCut];
        double relative = heuristic == VcmEqualMis ? 0 : power * (score.Logarithm - largest);
        selected = VcmLogDensity(relative - VcmLog2Positive(sum), score.Valid);
    }
    [loop]
    for (uint cut = 0; cut <= count; ++cut)
    {
        result.Connection[cut] /= sum;
        result.Merging[cut] /= sum;
    }
    return VcmPackMis(result, selected);
}
VcmWeights VcmJointWeights(VcmVertexPdf vertices[VcmMaximumVertices], uint count,
                           VcmStrategy connections[VcmMaximumStrategies],
                           VcmStrategy mergings[VcmMaximumStrategies], double radius, uint heuristic,
                           uint families)
{
    VcmMisEvaluation evaluation = VcmEvaluateMis(vertices, count, connections, mergings, radius, heuristic,
                                                 families, VcmMaximumStrategies, 0);
    return evaluation.Weights;
}
