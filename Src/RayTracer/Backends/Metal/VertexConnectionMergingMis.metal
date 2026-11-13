// Independently authored Metal VCM path-space MIS. Include ExtendedFloat.metal
// first: Metal has no double type, and float denormals may be flushed (MSL
// specification, sections 2.1 and 8.1). Scaled two-float products retain about
// 48 significand bits and keep tiny weights usable by a later merging kernel.
// https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
// Georgiev et al. (2012), "Light Transport Simulation with Vertex Connection
// and Merging"; "Implementing Vertex Connection and Merging", revision 2.
// https://www.iliyan.com/publications/ImplementingVCM/ImplementingVCM_TechRep2012_rev2.pdf
//
// This implementation enumerates complete proposal products independently.
// O(n^2) work at bounded depth avoids dividing by a zero edge PDF, accumulated
// log cancellation, and assumptions that adjacent connection/merge cuts share
// eligibility. No transport evaluator from another integrator is included.
typedef WideMath::Number VcmNumber;
constant uint VcmMaximumReflections = 32;
constant uint VcmMaximumVertices = VcmMaximumReflections + 3;
constant uint VcmMaximumStrategies = VcmMaximumVertices + 1;
constant uint VcmDelta = 1, VcmNotIntersectable = 2, VcmVolumeScattering = 4;
constant uint VcmConnections = 1, VcmMergings = 2;
constant uint VcmEqual = 0, VcmBalance = 1, VcmPower = 2;
// Bounded input exponents leave ample signed-integer headroom for every
// product, its square, and normalization. Physical binary32 geometry is well
// inside this range; the representation is deliberately not IEEE binary64.
constant int VcmDensityExponentLimit = 1 << 20;
constant float VcmResidualLimit = 1.0f / 16777216.0f;
struct VcmVertexPdf
{
    VcmNumber Forward, Reverse;
    uint Flags, Reserved0, Reserved1, Reserved2;
};
struct VcmStrategy
{
    // All factors are per-strategy, including endpoint-only cuts. The caller
    // supplies one for an endpoint or source-selection factor it does not sample.
    VcmNumber LightEndpoint, CameraEndpoint, LightSelection, SampleCount;
    uint Enabled, Reserved0, Reserved1, Reserved2;
};
struct VcmWeights
{
    VcmNumber Connection[VcmMaximumStrategies], Merging[VcmMaximumStrategies];
    uint ActiveConnections, ActiveMergings, Reserved0, Reserved1;
};
VcmNumber VcmZero()
{
    return {0, 0, 0, 0};
}
VcmNumber VcmOne()
{
    return {0.5f, 0, 1, 0};
}
bool VcmPositive(VcmNumber value)
{
    return value.High > 0;
}
bool VcmUsableDensity(VcmNumber value)
{
    return value.High >= 0.5f && value.High < 1 && isfinite(value.Low) &&
           fabs(value.Low) <= VcmResidualLimit && value.Exponent >= -VcmDensityExponentLimit &&
           value.Exponent <= VcmDensityExponentLimit;
}
bool VcmFiniteScalar(float value)
{
    return (as_type<uint>(value) & ~WideMath::SignMask) < WideMath::InfinityBits;
}
VcmNumber VcmDensityFromFloat(float value)
{
    uint bits = as_type<uint>(value);
    return bits > 0 && bits < WideMath::InfinityBits ? WideMath::Decode(bits) : VcmZero();
}
VcmNumber VcmSquareRoot(VcmNumber value)
{
    if (!VcmPositive(value))
        return VcmZero();
    int exponent = value.Exponent / 2;
    int remainder = value.Exponent - exponent * 2;
    VcmNumber estimate = WideMath::Shift(WideMath::FromFloat(sqrt(ldexp(value.High, remainder))), exponent);
    // Two Newton steps refine the binary32 seed without ever materializing the
    // potentially unrepresentable magnitude in a native float.
    const uint Refinements = 2;
    for (uint iteration = 0; iteration < Refinements; ++iteration)
        estimate = WideMath::Shift(WideMath::Add(estimate, WideMath::Divide(value, estimate)), -1);
    return estimate;
}
struct VcmWideVector
{
    VcmNumber X, Y, Z;
};
VcmWideVector VcmDecodeVector(float3 value)
{
    return {WideMath::FromFloat(value.x), WideMath::FromFloat(value.y), WideMath::FromFloat(value.z)};
}
bool VcmFiniteVector(float3 value)
{
    return VcmFiniteScalar(value.x) && VcmFiniteScalar(value.y) && VcmFiniteScalar(value.z);
}
VcmNumber VcmDot(VcmWideVector a, VcmWideVector b)
{
    VcmNumber x = WideMath::Multiply(a.X, b.X), y = WideMath::Multiply(a.Y, b.Y),
              z = WideMath::Multiply(a.Z, b.Z);
    // Cancel large opposite products before adding a tiny grazing term.
    if (WideMath::Compare(WideMath::Absolute(y), WideMath::Absolute(x)) > 0)
    {
        VcmNumber swap = x;
        x = y;
        y = swap;
    }
    if (WideMath::Compare(WideMath::Absolute(z), WideMath::Absolute(y)) > 0)
    {
        VcmNumber swap = y;
        y = z;
        z = swap;
    }
    if (WideMath::Compare(WideMath::Absolute(y), WideMath::Absolute(x)) > 0)
    {
        VcmNumber swap = x;
        x = y;
        y = swap;
    }
    return WideMath::Add(WideMath::Add(x, y), z);
}
VcmNumber VcmSurfaceArrival(VcmNumber angular, float3 displacement, float3 geometricNormal)
{
    if (!VcmUsableDensity(angular) || !VcmFiniteVector(displacement) || !VcmFiniteVector(geometricNormal))
        return VcmZero();
    VcmWideVector segment = VcmDecodeVector(displacement);
    VcmNumber distanceSquared = VcmDot(segment, segment);
    VcmNumber projected = WideMath::Absolute(VcmDot(VcmDecodeVector(geometricNormal), segment));
    if (!VcmPositive(distanceSquared) || !VcmPositive(projected))
        return VcmZero();
    // p_A = p_w |n.d| / |d|^3, also for generalized delta branch masses.
    VcmNumber denominator = WideMath::Multiply(distanceSquared, VcmSquareRoot(distanceSquared));
    return WideMath::Divide(WideMath::Multiply(angular, projected), denominator);
}
VcmNumber VcmProjectedDiskArrival(VcmNumber disk, float3 geometricNormal, float3 unitDirection)
{
    if (!VcmUsableDensity(disk) || !VcmFiniteVector(geometricNormal) || !VcmFiniteVector(unitDirection))
        return VcmZero();
    VcmNumber projected =
        WideMath::Absolute(VcmDot(VcmDecodeVector(geometricNormal), VcmDecodeVector(unitDirection)));
    // No inverse-square falloff for an infinite emitter's first hit.
    return WideMath::Multiply(disk, projected);
}
float3 VcmUnitVector(float3 value)
{
    if (!VcmFiniteVector(value))
        return float3(0);
    VcmWideVector wide = VcmDecodeVector(value);
    VcmNumber length = VcmSquareRoot(VcmDot(wide, wide));
    if (!VcmPositive(length))
        return float3(0);
    return float3(as_type<float>(WideMath::Encode(WideMath::Divide(wide.X, length))),
                  as_type<float>(WideMath::Encode(WideMath::Divide(wide.Y, length))),
                  as_type<float>(WideMath::Encode(WideMath::Divide(wide.Z, length))));
}
VcmNumber VcmDiskArea(VcmNumber radius)
{
    if (!VcmUsableDensity(radius))
        return VcmZero();
    const float PiHigh = 3.14159265358979323846f;
    const float PiResidual = -8.7422780003724856617e-8f;
    VcmNumber pi = WideMath::Normalize(PiHigh, PiResidual, 0);
    return WideMath::Multiply(pi, WideMath::Multiply(radius, radius));
}
VcmNumber VcmSphereVolume(VcmNumber radius)
{
    if (!VcmUsableDensity(radius)) return VcmZero();
    const float PiHigh = 3.14159265358979323846f;
    const float PiResidual = -8.7422780003724856617e-8f;
    const VcmNumber pi = WideMath::Normalize(PiHigh, PiResidual, 0);
    return WideMath::Multiply(WideMath::Multiply(pi, WideMath::FromFloat(4.0f / 3.0f)),
                              WideMath::Multiply(radius, WideMath::Multiply(radius, radius)));
}
VcmNumber VcmIncludeDensity(VcmNumber product, VcmNumber factor)
{
    return VcmUsableDensity(factor) ? WideMath::Multiply(product, factor) : VcmZero();
}
VcmNumber VcmProposal(thread const VcmVertexPdf* vertices, uint count, VcmStrategy strategy, uint cut,
                      bool merging, VcmNumber area, VcmNumber volume)
{
    if (strategy.Enabled == 0 || !VcmUsableDensity(strategy.SampleCount) ||
        !VcmUsableDensity(strategy.LightSelection) ||
        WideMath::Compare(strategy.LightSelection, VcmOne()) > 0)
        return VcmZero();
    if (merging)
    {
        if (cut < 2 || cut >= count || !VcmPositive(area) || (vertices[cut - 1].Flags & VcmDelta) != 0)
            return VcmZero();
    }
    else if ((cut == 0 && (vertices[0].Flags & VcmNotIntersectable) != 0) ||
             (cut == count && (vertices[count - 1].Flags & VcmNotIntersectable) != 0) ||
             (cut > 0 && cut < count &&
              ((cut > 1 && (vertices[cut - 1].Flags & VcmDelta) != 0) ||
               (cut + 1 < count && (vertices[cut].Flags & VcmDelta) != 0))))
        return VcmZero();

    VcmNumber proposal = strategy.SampleCount;
    proposal = VcmIncludeDensity(proposal, strategy.LightEndpoint);
    proposal = VcmIncludeDensity(proposal, strategy.LightSelection);
    proposal = VcmIncludeDensity(proposal, strategy.CameraEndpoint);
    // Canonical vertices x[0] ... x[n-1] run from source to sensor. A VC cut s
    // samples F[1..s-1] and R[s..n-2]; the endpoint-only strategies extend these
    // ranges to F[n-1] or R[0]. A merge additionally samples physical R[s-1].
    for (uint vertex = 1; vertex < cut; ++vertex)
        proposal = VcmIncludeDensity(proposal, vertices[vertex].Forward);
    for (uint vertex = cut; vertex + 1 < count; ++vertex)
        proposal = VcmIncludeDensity(proposal, vertices[vertex].Reverse);
    if (merging)
    {
        proposal = VcmIncludeDensity(proposal, vertices[cut - 1].Reverse);
        proposal = WideMath::Multiply(proposal,
            (vertices[cut - 1].Flags & VcmVolumeScattering) != 0 ? volume : area);
    }
    return proposal;
}
VcmWeights VcmJointWeights(thread const VcmVertexPdf* vertices, uint count,
                           thread const VcmStrategy* connections, thread const VcmStrategy* mergings,
                           VcmNumber radius, uint heuristic, uint families)
{
    VcmWeights weights = {};
    if (count < 2 || count > VcmMaximumVertices || heuristic > VcmPower ||
        (families & ~(VcmConnections | VcmMergings)) != 0 ||
        (!VcmUsableDensity(radius) && !(radius.High == 0 && radius.Low == 0)))
        return weights;
    VcmNumber sum = VcmZero(), area = VcmDiskArea(radius), volume = VcmSphereVolume(radius);
    for (uint cut = 0; cut <= count; ++cut)
    {
        VcmNumber vc = (families & VcmConnections) != 0
                           ? VcmProposal(vertices, count, connections[cut], cut, false, area, volume)
                           : VcmZero();
        VcmNumber vm = (families & VcmMergings) != 0
                           ? VcmProposal(vertices, count, mergings[cut], cut, true, area, volume)
                           : VcmZero();
        weights.ActiveConnections += VcmPositive(vc) ? 1u : 0u;
        weights.ActiveMergings += VcmPositive(vm) ? 1u : 0u;
        if (heuristic == VcmEqual)
        {
            vc = VcmPositive(vc) ? VcmOne() : VcmZero();
            vm = VcmPositive(vm) ? VcmOne() : VcmZero();
        }
        else if (heuristic == VcmPower)
        {
            vc = WideMath::Multiply(vc, vc);
            vm = WideMath::Multiply(vm, vm);
        }
        weights.Connection[cut] = vc;
        weights.Merging[cut] = vm;
        sum = WideMath::Add(sum, WideMath::Add(vc, vm));
    }
    if (!VcmPositive(sum))
        return weights;
    for (uint cut = 0; cut <= count; ++cut)
    {
        if (VcmPositive(weights.Connection[cut]))
            weights.Connection[cut] = WideMath::Divide(weights.Connection[cut], sum);
        if (VcmPositive(weights.Merging[cut]))
            weights.Merging[cut] = WideMath::Divide(weights.Merging[cut], sum);
    }
    return weights;
}
