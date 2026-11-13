// Native DXR ray instancing. Software BLASes share object-local geometry; the
// hardware path validates candidate object-ray ranges before fixed-function
// traversal. Inverse-transpose normals and virtual primitive IDs are common
// surface contracts, not another backend's numerical implementation.
// PBRT 4e, "Applying Transformations"; DXR Functional Spec, "InstanceID".
// Integer transport retains subnormal intervals and restored distances across
// traversal calls even when float shader arithmetic can flush them to zero.
// https://github.com/microsoft/DirectXShaderCompiler/wiki/Denorm-Mode
struct InstanceGeometry
{
    uint NodeBytes, NodeCount, ReferenceOffset, TriangleOffset;
    uint SphereOffset, TriangleCount, FirstPrimitive, ObjectIndex;
    float3 Minimum;
    uint SphereCount;
    float3 Maximum;
    uint Padding;
};
struct InstanceRecord
{
    double4 Inverse0, Inverse1, Inverse2;
    InstanceGeometry Geometry;
    float3 Minimum;
    uint Padding0;
    float3 Maximum;
    uint Padding1;
    MotionRigid Motion;
};
struct InstanceNode
{
    float3 Minimum;
    uint FirstOrEscape;
    float3 Maximum;
    uint Count;
};
StructuredBuffer<InstanceRecord> InstanceRecords : register(t11);
StructuredBuffer<InstanceNode> InstanceNodes : register(t12);
RWByteAddressBuffer InstanceErrors : register(u4);
static const uint InstanceRangeError = 1u;
static const uint TieSearchFlag = 0x80000000u;
static const uint FloatSignBit = 0x80000000u, FloatMinNormal = 0x00800000u;
static const uint FloatInfinity = 0x7f800000u, DoubleExponent = 0x7ff00000u;
static const uint DoubleBias = 1023u, FloatSubnormalPower = 149u, DoubleHighMantissa = 20u;
static const uint DoubleMantissaBits = 52u;
static const double FloatLargest = 3.40282346638528859811704183484516925440e38L;
static const double FloatSmallestNormal = 1.17549435082228750796873653722224568e-38L;
double WideRayBits(uint encoding)
{
    const uint magnitude = encoding & ~FloatSignBit;
    if (magnitude >= FloatMinNormal) return double(asfloat(encoding));
    if (magnitude == 0u) return 0.0L;
    uint highest = firstbithigh(magnitude);
    uint fraction = magnitude ^ (1u << highest);
    uint low = highest > DoubleHighMantissa ? fraction << (DoubleMantissaBits - highest) : 0u;
    uint highFraction = highest > DoubleHighMantissa ? fraction >> (highest - DoubleHighMantissa)
                                                     : fraction << (DoubleHighMantissa - highest);
    uint exponent = DoubleBias + highest - FloatSubnormalPower;
    return asdouble(low, (encoding & FloatSignBit) | (exponent << DoubleHighMantissa) | highFraction);
}
uint RayParameterBits(double value)
{
    if (value > FloatLargest) return FloatInfinity;
    if (value >= FloatSmallestNormal) return asuint(float(value));
    uint low, high;
    asuint(value, low, high);
    // Adjusting the binary exponent scales by 2^149 exactly before integer
    // rounding. A zero input still rounds to zero in the subnormal branch.
    const double position = asdouble(low, high + (FloatSubnormalPower << DoubleHighMantissa));
    uint lower = uint(position);
    const double remainder = position - double(lower);
    if (remainder > 0.5L || (remainder == 0.5L && (lower & 1u) != 0u)) ++lower;
    return lower;
}
double AbsoluteWide(double value)
{
    return value < 0.0L ? -value : value;
}
double MinimumWide(double a, double b)
{
    return a < b ? a : b;
}
double MaximumWide(double a, double b)
{
    return a > b ? a : b;
}
double3 WideRayVector(uint3 value)
{
    return double3(WideRayBits(value.x), WideRayBits(value.y), WideRayBits(value.z));
}
double LargestComponent(double3 value)
{
    return MaximumWide(AbsoluteWide(value.x), MaximumWide(AbsoluteWide(value.y), AbsoluteWide(value.z)));
}
double DirectionFactor(double3 direction)
{
    static const double ArithmeticHeadroom = 4.0L;
    double largest = LargestComponent(direction);
    if (largest >= FloatSmallestNormal * ArithmeticHeadroom && largest <= FloatLargest / ArithmeticHeadroom)
        return 1.0L;
    uint low, high;
    asuint(largest, low, high);
    return asdouble(0u, high & DoubleExponent);
}
bool InstanceInterval(float3 minimum, float3 maximum, EncodedRay ray, uint farValue)
{
    double nearT = WideRayBits(ray.Minimum), farT = WideRayBits(farValue);
    for (uint axis = 0; axis < 3; ++axis)
    {
        double d = WideRayBits(ray.Direction[axis]), o = WideRayBits(ray.Origin[axis]);
        if (d == 0.0L)
        {
            if (o < double(minimum[axis]) || o > double(maximum[axis])) return false;
        }
        else
        {
            double a = (double(minimum[axis]) - o) / d, b = (double(maximum[axis]) - o) / d;
            nearT = MaximumWide(nearT, MinimumWide(a, b));
            farT = MinimumWide(farT, MaximumWide(a, b));
            if (nearT > farT) return false;
        }
    }
    return true;
}
float3 InstanceRestDirection(InstanceRecord instance, float3 direction)
{
    if (instance.Motion.Enabled == 0)
        return direction;
    const MotionPose pose = InterpolateMotion(instance.Motion.Transform, InstanceSceneTime());
    const double3 rest = InverseMotionCoordinates(pose, double3(direction));
    return float3(rest / LargestComponent(rest));
}
Payload InstanceSurface(Payload local, InstanceRecord instance, uint localPrimitive, uint3 worldDirection)
{
    if (instance.Motion.Enabled != 0)
    {
        const MotionPose pose = InterpolateMotion(instance.Motion.Transform, InstanceSceneTime());
        local.Normal = TransformMotionVector(pose, local.Normal);
        local.GeometricNormal = TransformMotionVector(pose, local.GeometricNormal);
    }
    double3 normal = instance.Inverse0.xyz * double(local.Normal.x) +
                     instance.Inverse1.xyz * double(local.Normal.y) +
                     instance.Inverse2.xyz * double(local.Normal.z);
    local.Normal = normalize(float3(normal / LargestComponent(normal)));
    double3 geometric = instance.Inverse0.xyz * double(local.GeometricNormal.x) +
                        instance.Inverse1.xyz * double(local.GeometricNormal.y) +
                        instance.Inverse2.xyz * double(local.GeometricNormal.z);
    local.GeometricNormal = normalize(float3(geometric / LargestComponent(geometric)));
    // Keep the object's exterior under negative determinants. Object-ray sidedness
    // is unchanged by the inverse-transpose transform, for hardware and software hits.
    const double3 direction = WideRayVector(worldDirection);
    const double facing = double(local.Normal.x) * direction.x + double(local.Normal.y) * direction.y +
                          double(local.Normal.z) * direction.z;
    if (facing > 0.0L) local.Normal = -local.Normal;
    local.PrimitiveIndex = instance.Geometry.FirstPrimitive + localPrimitive;
    return local;
}
Payload CastSoftwareObject(float3 origin, float3 direction, float minimumDistance, float maximumDistance,
                           InstanceGeometry geometry);

Payload TraverseInstances(EncodedRay ray, bool validateHardware, out bool valid)
{
    Payload closest = MissEncodedPayload(ray.Maximum);
    valid = true;
    uint cursor = 0;
    double3 worldOrigin = WideRayVector(ray.Origin), worldDirection = WideRayVector(ray.Direction);
    while (cursor < InstanceNodeCount)
    {
        InstanceNode node = InstanceNodes[cursor];
        if (!InstanceInterval(node.Minimum, node.Maximum, ray, closest.DistanceBits))
        {
            cursor = node.Count == 0 ? node.FirstOrEscape : cursor + 1;
            continue;
        }
        if (node.Count == 0)
        {
            ++cursor;
            continue;
        }
        for (uint item = 0; item < node.Count; ++item)
        {
            InstanceRecord instance = InstanceRecords[node.FirstOrEscape + item];
            if (!InstanceInterval(instance.Minimum, instance.Maximum, ray, closest.DistanceBits)) continue;
            double3 localOrigin, localDirection;
            for (uint row = 0; row < 3; ++row)
            {
                double4 inverse =
                    row == 0 ? instance.Inverse0 : (row == 1 ? instance.Inverse1 : instance.Inverse2);
                localOrigin[row] = inverse.w;
                localDirection[row] = 0.0L;
                for (uint column = 0; column < 3; ++column)
                {
                    localOrigin[row] += inverse[column] * worldOrigin[column];
                    localDirection[row] += inverse[column] * worldDirection[column];
                }
            }
            if (instance.Motion.Enabled != 0)
            {
                const MotionPose pose = InterpolateMotion(instance.Motion.Transform, InstanceSceneTime());
                InverseInstanceMotion(pose, localOrigin, localDirection);
            }
            double magnitude = LargestComponent(localDirection);
            if (LargestComponent(localOrigin) > FloatLargest || magnitude == 0.0L ||
                (validateHardware && (magnitude > FloatLargest || magnitude < FloatSmallestNormal)))
            {
                uint ignored;
                InstanceErrors.InterlockedOr(0u, InstanceRangeError, ignored);
                valid = false;
                continue;
            }
            if (validateHardware) continue;
            double factor = DirectionFactor(localDirection);
            Payload candidate = CastSoftwareObject(
                float3(localOrigin), float3(localDirection / factor),
                asfloat(RayParameterBits(WideRayBits(ray.Minimum) * factor)),
                asfloat(RayParameterBits(WideRayBits(ray.Maximum) * factor)), instance.Geometry);
            if ((candidate.Flags & SurfaceHitFlag) == 0u) continue;
            candidate.DistanceBits = RayParameterBits(WideRayBits(candidate.DistanceBits) / factor);
            uint distance = candidate.DistanceBits;
            if (distance <= ray.Minimum || distance >= ray.Maximum || distance > closest.DistanceBits)
                continue;
            candidate = InstanceSurface(candidate, instance, candidate.PrimitiveIndex, ray.Direction);
            if (distance == closest.DistanceBits && candidate.PrimitiveIndex >= closest.PrimitiveIndex)
                continue;
            closest = candidate;
        }
        ++cursor;
    }
    return closest;
}
