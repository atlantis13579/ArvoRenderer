// Native CUDA two-level traversal. The TLAS selects an affine ray instance;
// the selected CUDA BLAS traverses shared object-space geometry. Translation
// affects origins only. Inverse-transpose normals are normalized in world space.
// Extreme directions use a power-of-two parameter change and restore world t.
// PBRT (4th ed.), "Applying Transformations" and "Transformed Primitives".
// https://pbr-book.org/4ed/Geometry_and_Transformations/Applying_Transformations

template <class Bounds> __device__ bool InstanceBounds(const Bounds &node, const Ray &ray, float closest)
{
    double near = ray.MinimumDistance, far = fminf(ray.MaximumDistance, closest);
    for (unsigned int axis = 0; axis < 3; ++axis)
    {
        const double origin = Component(ray.Origin, axis), direction = Component(ray.Direction, axis);
        if (direction == 0)
        {
            if (origin < node.Minimum[axis] || origin > node.Maximum[axis])
                return false;
        }
        else
        {
            const double first = (double(node.Minimum[axis]) - origin) / direction;
            const double second = (double(node.Maximum[axis]) - origin) / direction;
            near = fmax(near, fmin(first, second));
            far = fmin(far, fmax(first, second));
            if (near > far)
                return false;
        }
    }
    return true;
}

__device__ float InstanceDistance(float distance, double scale)
{
    const double value = double(distance) * scale;
    return value > FLT_MAX ? CUDART_INF_F : float(value);
}

__device__ bool IntersectInstances(const PackedAccelerationNode *nodes, const PackedPrimitiveRef *references,
                                   const PackedTriangle *triangles, const PackedSphere *spheres,
                                   const FrameConstants &constants, const Ray &ray, bool anyHit, Hit &hit,
                                   unsigned long long &tests, float sceneTime)
{
    using Arvo::RayTracing::CudaInstances::InvalidObjectRay;
    bool accepted = false;
    hit.PrimitiveIndex = InvalidPrimitiveIndex;
    unsigned int cursor = 0;
    while (cursor < constants.InstanceNodeCount)
    {
        const auto node = constants.InstanceNodes[cursor];
        if (!InstanceBounds(node, ray, hit.Distance))
        {
            cursor = node.Count ? cursor + 1 : node.FirstOrEscape;
            continue;
        }
        if (!node.Count)
        {
            ++cursor;
            continue;
        }
        for (unsigned int leaf = 0; leaf < node.Count; ++leaf)
        {
            const auto &instance = constants.InstanceRecords[node.FirstOrEscape + leaf];
            if (!InstanceBounds(instance, ray, hit.Distance))
                continue;
            double localOrigin[3], localDirection[3];
            bool representable = true;
            double largestDirection = 0;
            for (unsigned int row = 0; row < 3; ++row)
            {
                const double *inverse = instance.WorldToObject + row * 4;
                localOrigin[row] = inverse[3];
                localDirection[row] = 0;
                for (unsigned int column = 0; column < 3; ++column)
                {
                    localOrigin[row] += inverse[column] * double(Component(ray.Origin, column));
                    localDirection[row] += inverse[column] * double(Component(ray.Direction, column));
                }
            }
            MotionPose pose{};
            if (instance.Motion.Enabled)
            {
                pose = InterpolateMotion(InstanceMotionTrack(instance.Motion.Transform), sceneTime);
                InverseInstanceMotion(pose, localOrigin, localDirection);
            }
            for (unsigned int axis = 0; axis < 3; ++axis)
            {
                representable &= isfinite(localOrigin[axis]) && fabs(localOrigin[axis]) <= FLT_MAX;
                largestDirection = fmax(largestDirection, fabs(localDirection[axis]));
            }
            if (!representable || largestDirection == 0 || !isfinite(largestDirection))
            {
                atomicOr(constants.InstanceError, InvalidObjectRay);
                continue;
            }
            constexpr double DirectionHeadroom = 4;
            double parameterScale = 1;
            if (largestDirection > FLT_MAX / DirectionHeadroom ||
                largestDirection < FLT_MIN * DirectionHeadroom)
            {
                int exponent;
                frexp(largestDirection, &exponent);
                parameterScale = ldexp(1.0, exponent);
            }
            Ray localRay;
            localRay.Origin = {float(localOrigin[0]), float(localOrigin[1]), float(localOrigin[2])};
            localRay.Direction = {float(localDirection[0] / parameterScale),
                                  float(localDirection[1] / parameterScale),
                                  float(localDirection[2] / parameterScale)};
            localRay.MinimumDistance = InstanceDistance(ray.MinimumDistance, parameterScale);
            localRay.MaximumDistance = InstanceDistance(ray.MaximumDistance, parameterScale);
            FrameConstants object = constants;
            object.NodeCount = instance.NodeCount;
            object.TriangleCount = instance.TriangleCount;
            Hit candidate{};
            candidate.Distance = InstanceDistance(hit.Distance, parameterScale);
            candidate.PrimitiveIndex = InvalidPrimitiveIndex;
            const PackedTriangle *objectTriangles = triangles ? triangles + instance.TriangleOffset : nullptr;
            const PackedSphere *objectSpheres = spheres ? spheres + instance.SphereOffset : nullptr;
            if (!IntersectGeometry(nodes + instance.NodeOffset, references + instance.ReferenceOffset,
                                   objectTriangles, objectSpheres, object, localRay, anyHit, candidate,
                                   tests, sceneTime))
                continue;
            // Any-hit traversal is allowed to return without surface attributes.
            if (anyHit)
                return true;
            const double worldDistance = double(candidate.Distance) / parameterScale;
            if (!isfinite(worldDistance) || worldDistance > hit.Distance)
                continue;
            candidate.Distance = float(worldDistance);
            if (candidate.Distance <= ray.MinimumDistance || candidate.Distance >= ray.MaximumDistance)
                continue;
            candidate.PrimitiveIndex += instance.FirstPrimitive;
            if (candidate.Distance == hit.Distance && candidate.PrimitiveIndex >= hit.PrimitiveIndex)
                continue;
            if (instance.Motion.Enabled)
            {
                const auto n = TransformMotionVector(pose, make_float3(candidate.Normal.X, candidate.Normal.Y, candidate.Normal.Z));
                const auto g = TransformMotionVector(pose, make_float3(candidate.GeometricNormal.X, candidate.GeometricNormal.Y, candidate.GeometricNormal.Z));
                candidate.Normal = {n.x, n.y, n.z};
                candidate.GeometricNormal = {g.x, g.y, g.z};
            }
            double normal[3]{};
            for (unsigned int row = 0; row < 3; ++row)
                for (unsigned int column = 0; column < 3; ++column)
                    normal[row] += instance.WorldToObject[column * 4 + row] *
                                   double(Component(candidate.Normal, column));
            const double length = norm3d(normal[0], normal[1], normal[2]);
            candidate.Normal = {float(normal[0] / length), float(normal[1] / length),
                                float(normal[2] / length)};
            double geometric[3]{};
            for (unsigned int component = 0; component < 3; ++component)
                for (unsigned int source = 0; source < 3; ++source)
                    geometric[component] += instance.WorldToObject[source * 4 + component] *
                                            double(Component(candidate.GeometricNormal, source));
            const double geometricLength = norm3d(geometric[0], geometric[1], geometric[2]);
            candidate.GeometricNormal = {float(geometric[0] / geometricLength),
                                         float(geometric[1] / geometricLength),
                                         float(geometric[2] / geometricLength)};
            // FrontFace is invariant under affine ray/normal dual transforms.
            // A reflected instance retains the object's exterior; ng is never flipped.
            const double facing = double(candidate.Normal.X) * ray.Direction.X +
                                  double(candidate.Normal.Y) * ray.Direction.Y +
                                  double(candidate.Normal.Z) * ray.Direction.Z;
            if (facing > 0)
                candidate.Normal = Multiply(candidate.Normal, -1.0f);
            hit = candidate;
            accepted = true;
        }
        ++cursor;
    }
    return accepted;
}
