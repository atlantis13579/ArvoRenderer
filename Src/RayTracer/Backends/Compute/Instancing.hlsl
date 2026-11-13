// Native Compute two-level traversal. Ray transforms preserve t; an exceptional
// direction scale is removed by an exact power of two and restored after BLAS
// traversal. Normal transport uses the inverse transpose, including reflections.
// HLSL float arithmetic may flush subnormals. Decode input ray components and
// encode subnormal output distances explicitly at the double/float boundary.
// PBRT 4th ed., "Applying Transformations" and "Transformed Primitives".
// https://pbr-book.org/4ed/Geometry_and_Transformations/Applying_Transformations
struct InstanceRecord
{
    double4 InverseX, InverseY, InverseZ;
    GeometryRange Geometry;
    float3 Minimum;
    uint ObjectIndex;
    float3 Maximum;
    uint Padding;
    MotionRigid Motion;
};
StructuredBuffer<InstanceRecord> Instances : register(t9);
StructuredBuffer<PackedBvhNode> InstanceNodes : register(t10);
RWByteAddressBuffer InstanceError : register(u6);
static const uint FloatSign = 0x80000000u, FloatMinimumNormalBits = 0x00800000u;
static const uint FloatPositiveInfinityBits = 0x7f800000u;
static const uint DoubleExponentMask = 0x7ff00000u, DoubleMantissaHighBits = 20u;
static const uint DoubleExponentBias = 1023u, FloatSubnormalExponent = 149u;
static const uint InvalidObjectRay = 1u;
static const double InstanceFloatMaximum = 3.40282346638528859811704183484516925440e38L;
static const double InstanceFloatMinimum = 1.17549435082228750796873653722224568e-38L;

double InstanceAbs(double x)
{
    return x < 0.0L ? -x : x;
}
double InstanceMin(double a, double b)
{
    return a < b ? a : b;
}
double InstanceMax(double a, double b)
{
    return a > b ? a : b;
}
double InstanceWiden(float value)
{
    const uint bits = asuint(value), magnitude = bits & ~FloatSign;
    if (magnitude >= FloatMinimumNormalBits)
        return double(value);
    const double quantum =
        asdouble(0u, (DoubleExponentBias - FloatSubnormalExponent) << DoubleMantissaHighBits);
    const double decoded = double(magnitude) * quantum;
    return (bits & FloatSign) != 0u ? -decoded : decoded;
}
float InstanceDistance(double distance)
{
    if (distance > InstanceFloatMaximum)
        return asfloat(FloatPositiveInfinityBits);
    if (distance >= InstanceFloatMinimum)
        return float(distance);
    const double quantum =
        asdouble(0u, (DoubleExponentBias - FloatSubnormalExponent) << DoubleMantissaHighBits);
    const double units = distance / quantum;
    uint mantissa = uint(units);
    const double fraction = units - double(mantissa);
    // Match round-to-nearest, ties-to-even without a float arithmetic operation.
    if (fraction > 0.5L || (fraction == 0.5L && (mantissa & 1u) != 0u))
        ++mantissa;
    return asfloat(mantissa);
}
bool InstanceBox(float3 minimum, float3 maximum, RayData ray, float closest)
{
    double nearT = InstanceWiden(ray.MinimumDistance);
    double farT = InstanceMin(InstanceWiden(ray.MaximumDistance), InstanceWiden(closest));
    [unroll] for (uint axis = 0u; axis < 3u; ++axis)
    {
        const double origin = InstanceWiden(ray.Origin[axis]), direction = InstanceWiden(ray.Direction[axis]);
        if (direction == 0.0L)
        {
            if (origin < double(minimum[axis]) || origin > double(maximum[axis]))
                return false;
        }
        else
        {
            const double first = (double(minimum[axis]) - origin) / direction;
            const double second = (double(maximum[axis]) - origin) / direction;
            nearT = InstanceMax(nearT, InstanceMin(first, second));
            farT = InstanceMin(farT, InstanceMax(first, second));
            if (nearT > farT)
                return false;
        }
    }
    return true;
}
bool IntersectInstances(RayData ray, bool anyHit, inout HitData hit, inout uint2 primitiveTests, float sceneTime)
{
    bool found = false;
    uint cursor = 0u;
    hit.PrimitiveIndex = InvalidPrimitiveIndex;
    const double3 worldOrigin =
        double3(InstanceWiden(ray.Origin.x), InstanceWiden(ray.Origin.y), InstanceWiden(ray.Origin.z));
    const double3 worldDirection = double3(InstanceWiden(ray.Direction.x), InstanceWiden(ray.Direction.y),
                                           InstanceWiden(ray.Direction.z));
    [loop] while (cursor < InstanceNodeCount)
    {
        const PackedBvhNode node = InstanceNodes[cursor];
        if (!InstanceBox(node.Minimum, node.Maximum, ray, hit.Distance))
        {
            cursor = node.PrimitiveCount == 0u ? node.FirstOrEscape : cursor + 1u;
            continue;
        }
        if (node.PrimitiveCount == 0u)
        {
            ++cursor;
            continue;
        }
        [loop] for (uint leaf = 0u; leaf < node.PrimitiveCount; ++leaf)
        {
            const InstanceRecord instance = Instances[node.FirstOrEscape + leaf];
            if (!InstanceBox(instance.Minimum, instance.Maximum, ray, hit.Distance))
                continue;
            double3 origin, direction;
            [unroll] for (uint row = 0u; row < 3u; ++row)
            {
                const double4 inverse =
                    row == 0u ? instance.InverseX : (row == 1u ? instance.InverseY : instance.InverseZ);
                origin[row] = inverse.w;
                direction[row] = 0.0L;
                [unroll] for (uint column = 0u; column < 3u; ++column)
                {
                    origin[row] += inverse[column] * worldOrigin[column];
                    direction[row] += inverse[column] * worldDirection[column];
                }
            }
            MotionPose pose = (MotionPose)0;
            if (instance.Motion.Enabled != 0)
            {
                pose = InterpolateMotion(instance.Motion.Transform, sceneTime);
                InverseInstanceMotion(pose, origin, direction);
            }
            const double largest = InstanceMax(
                InstanceAbs(direction.x), InstanceMax(InstanceAbs(direction.y), InstanceAbs(direction.z)));
            if (InstanceAbs(origin.x) > InstanceFloatMaximum ||
                InstanceAbs(origin.y) > InstanceFloatMaximum ||
                InstanceAbs(origin.z) > InstanceFloatMaximum || largest == 0.0L)
            {
                uint ignored;
                InstanceError.InterlockedOr(0u, InvalidObjectRay, ignored);
                continue;
            }
            static const double DirectionHeadroom = 4.0L;
            double scale = 1.0L;
            if (largest > InstanceFloatMaximum / DirectionHeadroom ||
                largest < InstanceFloatMinimum * DirectionHeadroom)
            {
                uint low, high;
                asuint(largest, low, high);
                scale = asdouble(0u, high & DoubleExponentMask);
            }
            RayData local;
            local.Origin = float3(origin);
            local.Direction = float3(direction / scale);
            local.MinimumDistance = InstanceDistance(InstanceWiden(ray.MinimumDistance) * scale);
            local.MaximumDistance = InstanceDistance(InstanceWiden(ray.MaximumDistance) * scale);
            HitData candidate = (HitData)0;
            candidate.Distance = InstanceDistance(InstanceWiden(hit.Distance) * scale);
            candidate.PrimitiveIndex = InvalidPrimitiveIndex;
            if (!IntersectGeometry(local, instance.Geometry, anyHit, candidate, primitiveTests, sceneTime))
                continue;
            if (anyHit)
                return true; // Any-hit BLAS calls need not populate attributes.
            const double worldT = InstanceWiden(candidate.Distance) / scale;
            if (worldT <= InstanceWiden(ray.MinimumDistance) ||
                worldT >= InstanceWiden(ray.MaximumDistance) || worldT > InstanceWiden(hit.Distance))
                continue;
            candidate.Distance = InstanceDistance(worldT);
            const double roundedT = InstanceWiden(candidate.Distance);
            if (roundedT <= InstanceWiden(ray.MinimumDistance) ||
                roundedT >= InstanceWiden(ray.MaximumDistance))
                continue;
            candidate.PrimitiveIndex += instance.Geometry.FirstPrimitive;
            if (asuint(candidate.Distance) == asuint(hit.Distance) &&
                candidate.PrimitiveIndex >= hit.PrimitiveIndex)
                continue;
            if (instance.Motion.Enabled != 0)
            {
                candidate.Normal = TransformMotionVector(pose, candidate.Normal);
                candidate.GeometricNormal = TransformMotionVector(pose, candidate.GeometricNormal);
            }
            double3 normal;
            [unroll] for (uint axis = 0u; axis < 3u; ++axis) normal[axis] =
                instance.InverseX[axis] * double(candidate.Normal.x) +
                instance.InverseY[axis] * double(candidate.Normal.y) +
                instance.InverseZ[axis] * double(candidate.Normal.z);
            const double normalScale =
                InstanceMax(InstanceAbs(normal.x), InstanceMax(InstanceAbs(normal.y), InstanceAbs(normal.z)));
            candidate.Normal = normalize(float3(normal / normalScale));
            double3 geometric;
            [unroll] for (uint component = 0u; component < 3u; ++component) geometric[component] =
                instance.InverseX[component] * double(candidate.GeometricNormal.x) +
                instance.InverseY[component] * double(candidate.GeometricNormal.y) +
                instance.InverseZ[component] * double(candidate.GeometricNormal.z);
            const double geometricScale = InstanceMax(InstanceAbs(geometric.x),
                InstanceMax(InstanceAbs(geometric.y), InstanceAbs(geometric.z)));
            candidate.GeometricNormal = normalize(float3(geometric / geometricScale));
            // The inverse transpose preserves object-side orientation even when
            // det(M) < 0; FrontFace from the conditioned object ray is invariant.
            const double facing = double(candidate.Normal.x) * worldDirection.x +
                                  double(candidate.Normal.y) * worldDirection.y +
                                  double(candidate.Normal.z) * worldDirection.z;
            if (facing > 0.0L)
                candidate.Normal = -candidate.Normal;
            hit = candidate;
            found = true;
        }
        ++cursor;
    }
    return found;
}
