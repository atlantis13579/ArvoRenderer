// Metal's own TLAS traversal and affine ray conditioning. A power-of-two change
// of direction scale preserves t exactly until the final binary32 rounding.
// Outward BLAS intervals are followed by the original open-interval test. Normals
// use the inverse transpose, including shear and nonuniformly scaled spheres.
// PBRT 4th ed., "Transformed Primitives"; Metal Shading Language Specification:
// https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
constant uint InstanceDimensions = 3u, InstanceAffineStride = 4u;
constant uint MinimumNormalBits = 0x00800000u;
constant uint MaximumFiniteBits = 0x7f7fffffu;
constant uint InstanceRangeError = 1u;

struct WideRay
{
    WideMath::Number Origin[3], Direction[3];
    WideMath::Number Minimum, Maximum;
};

WideRay DecodeRay(EncodedRay ray)
{
    WideRay result;
    for (uint axis = 0u; axis < InstanceDimensions; ++axis)
    {
        result.Origin[axis] = WideMath::Decode(ray.OriginMinimum[axis]);
        result.Direction[axis] = WideMath::Decode(ray.DirectionMaximum[axis]);
    }
    result.Minimum = WideMath::Decode(ray.OriginMinimum.w & ~WideMath::SignMask);
    result.Maximum = WideMath::Decode(ray.DirectionMaximum.w & ~WideMath::SignMask);
    return result;
}

bool InstanceBounds(uint3 minimum, uint3 maximum, WideRay ray, uint closest)
{
    WideMath::Number nearTime = ray.Minimum;
    // Retain candidates that round to the same public t but have a lower ID.
    const uint inclusiveClosest = closest < MaximumFiniteBits ? closest + 1u : closest;
    WideMath::Number farTime = WideMath::Minimum(ray.Maximum, WideMath::Decode(inclusiveClosest));
    for (uint axis = 0u; axis < InstanceDimensions; ++axis)
    {
        const auto lower = WideMath::Decode(minimum[axis]);
        const auto upper = WideMath::Decode(maximum[axis]);
        if (ray.Direction[axis].High == 0.0f)
        {
            if (WideMath::Compare(ray.Origin[axis], lower) < 0 ||
                WideMath::Compare(ray.Origin[axis], upper) > 0)
                return false;
        }
        else
        {
            const auto a = WideMath::Divide(WideMath::Subtract(lower, ray.Origin[axis]), ray.Direction[axis]);
            const auto b = WideMath::Divide(WideMath::Subtract(upper, ray.Origin[axis]), ray.Direction[axis]);
            nearTime = WideMath::Maximum(nearTime, WideMath::Minimum(a, b));
            farTime = WideMath::Minimum(farTime, WideMath::Maximum(a, b));
            if (WideMath::Compare(nearTime, farTime) > 0)
                return false;
        }
    }
    return true;
}

bool NativeCoordinate(WideMath::Number value, thread float &output)
{
    const uint bits = WideMath::Encode(value), magnitude = bits & ~WideMath::SignMask;
    // Primitive arithmetic uses native floats. Reject a lost component instead
    // of turning an unrepresentable affine ray into a plausible silent miss.
    if (magnitude > MaximumFiniteBits || (value.High != 0.0f && magnitude < MinimumNormalBits))
        return false;
    output = as_type<float>(bits);
    return true;
}

bool IntersectConditionedObject(ObjectGeometry object, WideRay local, EncodedRay original, bool anyHit,
                                thread HitData &hit, thread uint &distanceBits, thread uint2 &primitiveTests,
                                thread bool &valid)
{
    valid = true;
    if (object.NodeCount == 0u)
        return false;
    int exponent = 0;
    bool nonzero = false;
    for (uint axis = 0u; axis < InstanceDimensions; ++axis)
        if (local.Direction[axis].High != 0.0f)
        {
            exponent =
                nonzero ? max(exponent, local.Direction[axis].Exponent) : local.Direction[axis].Exponent;
            nonzero = true;
        }
    if (!nonzero)
    {
        valid = false;
        return false;
    }
    RayData ray;
    for (uint axis = 0u; axis < InstanceDimensions; ++axis)
    {
        float origin = 0.0f, direction = 0.0f;
        if (!NativeCoordinate(local.Origin[axis], origin) ||
            !NativeCoordinate(WideMath::Shift(local.Direction[axis], -exponent), direction))
        {
            valid = false;
            return false;
        }
        ray.Origin[axis] = origin;
        ray.Direction[axis] = direction;
    }

    const uint originalMin = original.OriginMinimum.w & ~WideMath::SignMask;
    const uint originalMax = original.DirectionMaximum.w & ~WideMath::SignMask;
    if (originalMin >= originalMax)
        return false;
    uint minimum = WideMath::Encode(WideMath::Shift(local.Minimum, exponent));
    uint maximum = WideMath::Encode(WideMath::Shift(local.Maximum, exponent));
    minimum = min(minimum, MaximumFiniteBits);
    if (minimum != 0u)
        --minimum;
    if (maximum < WideMath::InfinityBits)
        ++maximum;
    // Conservative native limits; integer checks below own acceptance.
    ray.MinimumDistance = minimum < MinimumNormalBits ? 0.0f : as_type<float>(minimum);
    ray.MaximumDistance = as_type<float>(max(maximum, MinimumNormalBits));
    for (;;)
    {
        HitData candidate{};
        candidate.Distance = ray.MaximumDistance;
        candidate.Color = float3(1.0f);
        if (!IntersectObject(object, ray, anyHit, candidate, primitiveTests))
            return false;
        const uint restored =
            WideMath::Encode(WideMath::Shift(WideMath::FromFloat(candidate.Distance), -exponent));
        if (restored <= originalMin)
        {
            // An outward lower bound can expose a root on the original endpoint.
            // Continue beyond it, retaining a sphere's otherwise valid far root.
            ray.MinimumDistance = candidate.Distance;
            continue;
        }
        if (restored >= originalMax)
            return false;
        hit = candidate;
        distanceBits = restored;
        hit.Distance = as_type<float>(restored);
        return true;
    }
}

WideRay TransformInstanceRay(device const InstanceRecord &instance, WideRay ray)
{
    WideRay local;
    local.Minimum = ray.Minimum;
    local.Maximum = ray.Maximum;
    for (uint row = 0u; row < InstanceDimensions; ++row)
    {
        const auto translation = instance.Inverse[row * InstanceAffineStride + InstanceDimensions];
        local.Origin[row] = WideMath::Normalize(translation.High, translation.Low, translation.Exponent);
        local.Direction[row] = WideMath::FromFloat(0.0f);
        for (uint column = 0u; column < InstanceDimensions; ++column)
        {
            const auto stored = instance.Inverse[row * InstanceAffineStride + column];
            const auto coefficient = WideMath::Normalize(stored.High, stored.Low, stored.Exponent);
            local.Origin[row] =
                WideMath::Add(local.Origin[row], WideMath::Multiply(coefficient, ray.Origin[column]));
            local.Direction[row] =
                WideMath::Add(local.Direction[row], WideMath::Multiply(coefficient, ray.Direction[column]));
        }
    }
    return local;
}

bool TransformInstanceNormal(device const InstanceRecord &instance, thread float3 &normal)
{
    WideMath::Number components[3];
    int exponent = 0;
    bool nonzero = false;
    for (uint row = 0u; row < InstanceDimensions; ++row)
    {
        components[row] = WideMath::FromFloat(0.0f);
        for (uint column = 0u; column < InstanceDimensions; ++column)
        {
            const auto stored = instance.Inverse[column * InstanceAffineStride + row];
            const auto coefficient = WideMath::Normalize(stored.High, stored.Low, stored.Exponent);
            components[row] = WideMath::Add(
                components[row], WideMath::Multiply(coefficient, WideMath::FromFloat(normal[column])));
        }
        if (components[row].High != 0.0f)
        {
            exponent = nonzero ? max(exponent, components[row].Exponent) : components[row].Exponent;
            nonzero = true;
        }
    }
    if (!nonzero)
        return false;
    for (uint axis = 0u; axis < InstanceDimensions; ++axis)
        normal[axis] = as_type<float>(WideMath::Encode(WideMath::Shift(components[axis], -exponent)));
    normal = normalize(normal);
    return all(isfinite(normal));
}

bool IntersectInstances(SceneGeometryBuffers scene, EncodedRay ray, bool anyHit, thread HitData &hit,
                        thread uint &distanceBits, thread uint2 &primitiveTests)
{
    const WideRay world = DecodeRay(ray);
    distanceBits = ray.DirectionMaximum.w & ~WideMath::SignMask;
    hit.PrimitiveIndex = InvalidPrimitiveIndex;
    bool found = false;
    uint nodeIndex = 0u;
    while (nodeIndex < scene.NodeCount)
    {
        const InstanceNode node = scene.Nodes[nodeIndex];
        if (!InstanceBounds(uint3(node.Minimum), uint3(node.Maximum), world, distanceBits))
        {
            nodeIndex = node.Count ? nodeIndex + 1u : node.FirstOrEscape;
            continue;
        }
        if (!node.Count)
        {
            ++nodeIndex;
            continue;
        }
        for (uint offset = 0u; offset < node.Count; ++offset)
        {
            device const InstanceRecord &instance = scene.Instances[node.FirstOrEscape + offset];
            if (!InstanceBounds(uint3(instance.Minimum), uint3(instance.Maximum), world, distanceBits))
                continue;
            const InstanceGeometryRange range = instance.Geometry;
            const ObjectGeometry object = {scene.Object.Nodes + range.NodeBegin,
                                           scene.Object.References + range.ReferenceBegin,
                                           scene.Object.Triangles + range.TriangleBegin,
                                           scene.Object.Spheres + range.SphereBegin,
                                           range.NodeCount,
                                           range.TriangleCount,
                                           scene.Object.Acceleration,
                                           scene.Object.SceneTime};
            HitData candidate{};
            uint candidateDistance = distanceBits;
            bool valid = true;
            WideRay localRay = TransformInstanceRay(instance, world);
            MotionPose pose{};
            if (instance.Motion.Enabled != 0)
            {
                pose = InterpolateMotion(instance.Motion.Transform, scene.Object.SceneTime);
                InverseInstanceMotion(pose, localRay.Origin, localRay.Direction);
            }
            const bool accepted =
                IntersectConditionedObject(object, localRay, ray, anyHit,
                                           candidate, candidateDistance, primitiveTests, valid);
            if (!valid)
            {
                atomic_store_explicit(scene.Error, InstanceRangeError, memory_order_relaxed);
                return false;
            }
            if (!accepted)
                continue;
            candidate.PrimitiveIndex += range.FirstPrimitive;
            if (candidateDistance > distanceBits ||
                (candidateDistance == distanceBits && candidate.PrimitiveIndex >= hit.PrimitiveIndex))
                continue;
            // Inverse transpose preserves the object's exterior and FrontFace,
            // including reflected transforms. Only the shading normal faces the ray.
            if (instance.Motion.Enabled != 0)
            {
                candidate.Normal = TransformMotionVector(pose, candidate.Normal);
                candidate.GeometricNormal = TransformMotionVector(pose, candidate.GeometricNormal);
            }
            if (!TransformInstanceNormal(instance, candidate.Normal) ||
                !TransformInstanceNormal(instance, candidate.GeometricNormal))
            {
                atomic_store_explicit(scene.Error, InstanceRangeError, memory_order_relaxed);
                return false;
            }
            hit = candidate;
            distanceBits = candidateDistance;
            found = true;
            if (anyHit)
                return true;
        }
        ++nodeIndex;
    }
    return found;
}
