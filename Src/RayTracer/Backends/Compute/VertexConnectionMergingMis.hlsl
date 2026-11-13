// Independent Compute VCM measures. Georgiev et al. (2012), "Light Transport
// Simulation with Vertex Connection and Merging"; "Implementing VCM", rev. 2,
// equations 10-12 and section 4.3:
// https://www.iliyan.com/publications/ImplementingVCM/ImplementingVCM_TechRep2012_rev2.pdf
//
// A completed path is ordered light -> camera. VC cut s samples every interior
// vertex once. VM cut s additionally samples x[s-1] from the camera walk, so
// its density contains Reverse[s-1] * pi*r^2. Count attempted light paths, not
// accepted photons. Delta branch masses and their Jacobians remain in densities;
// only eligibility excludes adjacent connections and the singular vertex's merge.
//
// For clarity this implementation explicitly scans each cut (at most 35
// vertices). No other integrator or backend provides its density enumeration.
// Compile with IEEE strictness (-Gis); do not reassociate common-factor removal.
static const uint VcmMaximumReflections = 32, VcmMaximumVertices = VcmMaximumReflections + 3;
static const uint VcmMaximumStrategies = VcmMaximumVertices + 1;
static const uint VcmDeltaScattering = 1, VcmEndpointNotIntersectable = 2,
                  VcmVolumeScattering = 4;
static const uint VcmEqualMis = 0, VcmBalanceMis = 1, VcmPowerMis = 2;
static const uint VcmConnections = 1, VcmMergings = 2;
static const double VcmPi = 3.14159265358979323846L, VcmDoubleMaximum = 1.7976931348623157e308L;
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
bool VcmFinite(double value)
{
    return value >= -VcmDoubleMaximum && value <= VcmDoubleMaximum;
}
double VcmAbsolute(double value)
{
    return value < 0 ? -value : value;
}

// HLSL log2/exp2 only expose float intrinsics. Split IEEE binary64 to preserve
// the exponent range, then use bounded double series on the reduced mantissa.
// https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-log2
// https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/asdouble
static const uint VcmExponentShift = 20, VcmExponentMask = 0x7ff, VcmMantissaMask = 0xfffff;
static const int VcmExponentBias = 1023, VcmMinimumNormalExponent = -1022, VcmUnderflowExponent = -1075;
static const uint VcmMantissaBits = 52, VcmLogSeriesTerms = 12, VcmExpSeriesOrder = 16;
static const double VcmSubnormalScale = 4503599627370496.0L;
static const double VcmSqrtTwo = 1.4142135623730950488L;
static const double VcmInverseLnTwo = 1.4426950408889634074L, VcmLnTwo = 0.6931471805599453094L;
double VcmLog2Positive(double value)
{
    uint low, high;
    asuint(value, low, high);
    int adjustment = 0;
    if (((high >> VcmExponentShift) & VcmExponentMask) == 0)
    {
        value *= VcmSubnormalScale;
        adjustment = -int(VcmMantissaBits);
        asuint(value, low, high);
    }
    int exponent = int((high >> VcmExponentShift) & VcmExponentMask) - VcmExponentBias + adjustment;
    double mantissa = asdouble(low, (high & VcmMantissaMask) | (uint(VcmExponentBias) << VcmExponentShift));
    if (mantissa > VcmSqrtTwo)
    {
        mantissa *= 0.5L;
        ++exponent;
    }
    const double z = (mantissa - 1) / (mantissa + 1), squared = z * z;
    double term = z, sum = z;
    for (uint i = 1; i < VcmLogSeriesTerms; ++i)
    {
        term *= squared;
        sum += term / double(2 * i + 1);
    }
    // |z| <= (sqrt(2)-1)/(sqrt(2)+1); the omitted odd-series tail is below
    // binary64 epsilon. Subnormals were normalized before any float conversion.
    return double(exponent) + 2 * VcmInverseLnTwo * sum;
}
double VcmRelativeExp2(double value)
{
    if (value >= 0)
        return 1;
    if (!(value >= VcmUnderflowExponent))
        return 0;
    const int exponent = int(value - 0.5L);
    const double x = (value - double(exponent)) * VcmLnTwo;
    double series = 1;
    for (uint i = VcmExpSeriesOrder; i > 0; --i)
        series = 1 + x * series / double(i);
    if (exponent >= VcmMinimumNormalExponent)
        return series * asdouble(0, uint(exponent + VcmExponentBias) << VcmExponentShift);
    // Scale the final product into the subnormal range only once.
    double lowScale =
        asdouble(0, uint(exponent - VcmMinimumNormalExponent + VcmExponentBias) << VcmExponentShift);
    return (series * lowScale) * asdouble(0, 1u << VcmExponentShift);
}
VcmDensity VcmLogDensity(double logarithm, uint positive)
{
    VcmDensity result;
    result.Log2Value = logarithm;
    result.Positive = positive;
    result.Reserved = 0;
    return result;
}
VcmDensity VcmDensityFromValue(double value)
{
    if (value > 0 && VcmFinite(value))
        return VcmLogDensity(VcmLog2Positive(value), 1);
    return VcmLogDensity(0, 0);
}
bool VcmPositive(VcmDensity value)
{
    return value.Positive != 0 && VcmFinite(value.Log2Value);
}
double VcmGeometryComponent(float value)
{
    // Float arithmetic/conversion may flush subnormal inputs on shader hardware.
    // Decode their integer significand before any numeric float operation.
    // https://learn.microsoft.com/en-us/windows/win32/direct3d11/floating-point-rules
    static const uint SignBit = 0x80000000u, MinimumNormalBits = 0x00800000u;
    static const uint SubnormalExponent = 149;
    const uint bits = asuint(value), magnitude = bits & ~SignBit;
    if (magnitude >= MinimumNormalBits)
        return double(value);
    const double quantum = asdouble(0, (VcmExponentBias - SubnormalExponent) << VcmExponentShift);
    const double decoded = double(magnitude) * quantum;
    return (bits & SignBit) != 0 ? -decoded : decoded;
}
VcmDensity VcmSurfaceArrival(VcmDensity direction, float3 segment, float3 geometry)
{
    const double x = VcmGeometryComponent(segment.x), y = VcmGeometryComponent(segment.y),
                 z = VcmGeometryComponent(segment.z);
    const double squared = x * x + y * y + z * z;
    const double projected = VcmAbsolute(x * VcmGeometryComponent(geometry.x) +
                                        y * VcmGeometryComponent(geometry.y) +
                                        z * VcmGeometryComponent(geometry.z));
    if (!VcmPositive(direction) || !(squared > 0 && projected > 0) || !VcmFinite(squared) ||
        !VcmFinite(projected))
        return VcmLogDensity(0, 0);
    // p_A = p_w * |Ng . displacement| / |displacement|^3, including delta PMFs.
    const double logarithm =
        direction.Log2Value + VcmLog2Positive(projected) - 1.5 * VcmLog2Positive(squared);
    return VcmLogDensity(logarithm, VcmFinite(logarithm) ? 1 : 0);
}
VcmDensity VcmDiskArrival(VcmDensity disk, float3 geometry, float3 direction)
{
    const double x = VcmGeometryComponent(direction.x), y = VcmGeometryComponent(direction.y),
                 z = VcmGeometryComponent(direction.z);
    const double squared = x * x + y * y + z * z;
    const double projected = VcmAbsolute(x * VcmGeometryComponent(geometry.x) +
                                        y * VcmGeometryComponent(geometry.y) +
                                        z * VcmGeometryComponent(geometry.z));
    if (!VcmPositive(disk) || !(squared > 0 && projected > 0) || !VcmFinite(squared) || !VcmFinite(projected))
        return VcmLogDensity(0, 0);
    // Infinite sources use projected disk area, with no inverse-square factor.
    // Normalizing the supplied direction in log space also accepts scaled rays.
    const double logarithm = disk.Log2Value + VcmLog2Positive(projected) - 0.5 * VcmLog2Positive(squared);
    return VcmLogDensity(logarithm, VcmFinite(logarithm) ? 1 : 0);
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
struct VcmScore
{
    double Logarithm;
    uint Valid;
};
VcmScore VcmAppend(VcmScore product, VcmDensity density, double reference)
{
    if (!VcmPositive(density))
        product.Valid = 0;
    if (product.Valid != 0)
    {
        precise double residual = density.Log2Value - reference;
        precise double accumulated = product.Logarithm + residual;
        product.Logarithm = accumulated;
        product.Valid = VcmFinite(product.Logarithm) ? 1 : 0;
    }
    return product;
}
VcmScore VcmIncludeEndpoints(VcmScore path, VcmStrategy strategy)
{
    if (strategy.Enabled == 0 || !(strategy.LightSelection > 0 && strategy.LightSelection <= 1) ||
        !(strategy.SampleCount > 0) || !VcmFinite(strategy.SampleCount))
        path.Valid = 0;
    path = VcmAppend(path, strategy.LightEndpoint, 0);
    path = VcmAppend(path, strategy.CameraEndpoint, 0);
    if (path.Valid != 0)
    {
        path.Logarithm += VcmLog2Positive(strategy.LightSelection) + VcmLog2Positive(strategy.SampleCount);
        path.Valid = VcmFinite(path.Logarithm) ? 1 : 0;
    }
    return path;
}
VcmWeights VcmJointWeights(VcmVertexPdf vertices[VcmMaximumVertices], uint count,
                           VcmStrategy connection[VcmMaximumStrategies],
                           VcmStrategy merging[VcmMaximumStrategies], double radius, uint heuristic,
                           uint families)
{
    VcmWeights weights;
    weights.ActiveConnections = weights.ActiveMergings = weights.Reserved0 = weights.Reserved1 = 0;
    uint activeConnection[VcmMaximumStrategies], activeMerge[VcmMaximumStrategies];
    for (uint i = 0; i < VcmMaximumStrategies; ++i)
    {
        weights.Connection[i] = weights.Merging[i] = 0;
        activeConnection[i] = activeMerge[i] = 0;
    }
    if (count < 2 || count > VcmMaximumVertices || heuristic > VcmPowerMis || !(radius >= 0) ||
        !VcmFinite(radius) || families > (VcmConnections | VcmMergings))
        return weights;
    double reference[VcmMaximumVertices];
    for (uint i = 1; i + 1 < count; ++i)
    {
        const bool f = VcmPositive(vertices[i].Forward), r = VcmPositive(vertices[i].Reverse);
        reference[i] = f ? vertices[i].Forward.Log2Value : (r ? vertices[i].Reverse.Log2Value : 0);
        if (f && r && vertices[i].Reverse.Log2Value > reference[i])
            reference[i] = vertices[i].Reverse.Log2Value;
    }
    const VcmDensity area = VcmDiskArea(radius), volume = VcmSphereVolume(radius);
    double maximum = 0;
    for (uint s = 0; s <= count; ++s)
    {
        VcmScore path = {0, 1};
        for (uint i = 1; i + 1 < count; ++i)
        {
            VcmDensity density = vertices[i].Reverse;
            if (i < s)
                density = vertices[i].Forward;
            path = VcmAppend(path, density, reference[i]);
        }
        if (s == 0)
            path = VcmAppend(path, vertices[0].Reverse, 0);
        if (s == count)
            path = VcmAppend(path, vertices[count - 1].Forward, 0);
        const bool cutAllowed = s == 0 ? (vertices[0].Flags & VcmEndpointNotIntersectable) == 0
                                : s == count
                                    ? (vertices[count - 1].Flags & VcmEndpointNotIntersectable) == 0
                                    : !((s > 1 && (vertices[s - 1].Flags & VcmDeltaScattering) != 0) ||
                                        (s + 1 < count && (vertices[s].Flags & VcmDeltaScattering) != 0));
        VcmScore vc = {0, 0}, vm = {0, 0};
        if (cutAllowed && (families & VcmConnections) != 0)
            vc = VcmIncludeEndpoints(path, connection[s]);
        if (s >= 2 && s < count && (vertices[s - 1].Flags & VcmDeltaScattering) == 0 &&
            (families & VcmMergings) != 0)
        {
            // This second arrival is physical acceptance, not a common factor.
            // VM eligibility does not inherit the VC enable flag at this slot.
            VcmDensity measure = area;
            if ((vertices[s - 1].Flags & VcmVolumeScattering) != 0)
                measure = volume;
            vm = VcmIncludeEndpoints(VcmAppend(VcmAppend(path, vertices[s - 1].Reverse, 0), measure, 0),
                                     merging[s]);
        }
        if (vc.Valid != 0)
        {
            maximum = weights.ActiveConnections + weights.ActiveMergings == 0 || vc.Logarithm > maximum
                          ? vc.Logarithm
                          : maximum;
            weights.Connection[s] = vc.Logarithm;
            activeConnection[s] = 1;
            ++weights.ActiveConnections;
        }
        if (vm.Valid != 0)
        {
            maximum = weights.ActiveConnections + weights.ActiveMergings == 0 || vm.Logarithm > maximum
                          ? vm.Logarithm
                          : maximum;
            weights.Merging[s] = vm.Logarithm;
            activeMerge[s] = 1;
            ++weights.ActiveMergings;
        }
    }
    if (weights.ActiveConnections + weights.ActiveMergings == 0)
        return weights;
    const double power = heuristic == VcmPowerMis ? 2 : 1;
    double sum = 0;
    for (uint s = 0; s <= count; ++s)
    {
        weights.Connection[s] =
            activeConnection[s] != 0
                ? (heuristic == VcmEqualMis ? 1 : VcmRelativeExp2(power * (weights.Connection[s] - maximum)))
                : 0;
        weights.Merging[s] =
            activeMerge[s] != 0
                ? (heuristic == VcmEqualMis ? 1 : VcmRelativeExp2(power * (weights.Merging[s] - maximum)))
                : 0;
        sum += weights.Connection[s] + weights.Merging[s];
    }
    for (uint s = 0; s <= count; ++s)
    {
        weights.Connection[s] /= sum;
        weights.Merging[s] /= sum;
    }
    return weights;
}
