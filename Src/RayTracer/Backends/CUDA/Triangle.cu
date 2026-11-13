    // Moller-Trumbore in the open (min,max) interval. Scaling only the local
    // direction avoids a determinant tolerance that changes with instance scale.
    __device__ bool IntersectTriangle(
        const PackedTriangle& triangle,
        const Ray& ray,
        Hit& hit)
    {
        const Vec3 position0 = ToVec3(triangle.Position0);
        const Vec3 edge1 = ToVec3(triangle.Edge1);
        const Vec3 edge2 = ToVec3(triangle.Edge2);
        const float directionScale = fmaxf(fabsf(ray.Direction.X), fmaxf(fabsf(ray.Direction.Y), fabsf(ray.Direction.Z)));
        const Vec3 scaledDirection = {ray.Direction.X / directionScale, ray.Direction.Y / directionScale, ray.Direction.Z / directionScale};
        const Vec3 determinantVector = Cross(scaledDirection, edge2);
        const float determinant = Dot(edge1, determinantVector);
        if (!isfinite(determinant)
            || fabsf(determinant) < MinimumTriangleDeterminant * sqrtf(Dot(scaledDirection, scaledDirection))) return false;
        const float inverseDeterminant = 1.0f / determinant;
        const Vec3 originOffset = Subtract(ray.Origin, position0);
        const float barycentric1 = Dot(originOffset, determinantVector) * inverseDeterminant;
        if (barycentric1 < 0.0f || barycentric1 > 1.0f) return false;
        const Vec3 crossValue = Cross(originOffset, edge1);
        const float barycentric2 = Dot(scaledDirection, crossValue) * inverseDeterminant;
        if (barycentric2 < 0.0f || barycentric1 + barycentric2 > 1.0f) return false;
        const float distance = (Dot(edge2, crossValue) * inverseDeterminant) / directionScale;
        if (!isfinite(distance) || distance <= ray.MinimumDistance || distance >= ray.MaximumDistance || distance > hit.Distance)
        {
            return false;
        }

        const float barycentric0 = 1.0f - barycentric1 - barycentric2;
        const Vec3 geometricNormal = Normalize(Cross(edge1, edge2));
        Vec3 normal = geometricNormal;
        if ((triangle.Flags & 1u) != 0)
        {
            const Vec3 smoothNormal = Add(
                Add(Multiply(ToVec3(triangle.Normal0), barycentric0),
                    Multiply(ToVec3(triangle.Normal1), barycentric1)),
                Multiply(ToVec3(triangle.Normal2), barycentric2));
            if (Dot(smoothNormal, smoothNormal) > MinimumDirectionLengthSquared) normal = Normalize(smoothNormal);
        }
        if (Dot(normal, ray.Direction) > 0.0f) normal = Multiply(normal, -1.0f);
        hit.Distance = distance;
        hit.Normal = normal;
        hit.GeometricNormal = geometricNormal;
        hit.FrontFace = Dot(geometricNormal, scaledDirection) < 0.0f;
        hit.Color = Add(
            Add(Multiply(ToVec3(triangle.Color0), barycentric0),
                Multiply(ToVec3(triangle.Color1), barycentric1)),
            Multiply(ToVec3(triangle.Color2), barycentric2));
        hit.MaterialIndex = triangle.MaterialIndex;
        hit.TexCoordX = triangle.TexCoord01.X * barycentric0
            + triangle.TexCoord01.Z * barycentric1
            + triangle.TexCoord2.X * barycentric2;
        hit.TexCoordY = triangle.TexCoord01.Y * barycentric0
            + triangle.TexCoord01.W * barycentric1
            + triangle.TexCoord2.Y * barycentric2;
        hit.HasTexCoord = 1u;
#if defined(ARVO_SPECTRAL_TRANSPORT)
        hit.VertexSpectrumBase = triangle.Padding0;
        hit.Barycentric = make_float2(barycentric1, barycentric2);
#endif
        return true;
    }
