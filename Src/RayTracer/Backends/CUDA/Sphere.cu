    // Solve a*t^2 + 2*b*t + c with a non-unit direction. A local scale keeps
    // a well conditioned; roots are converted back to the original ray t.
    // The q-form avoids cancellation; both bounds of the query interval are open.
    __device__ bool IntersectSphere(const PackedSphere& sphere, const Ray& ray, Hit& hit)
    {
        const Vec3 center = ToVec3(sphere.CenterRadius);
        const float radius = sphere.CenterRadius.W;
        const Vec3 offset = Subtract(ray.Origin, center);
        const float directionScale = fmaxf(fabsf(ray.Direction.X), fmaxf(fabsf(ray.Direction.Y), fabsf(ray.Direction.Z)));
        const Vec3 scaledDirection = {ray.Direction.X / directionScale, ray.Direction.Y / directionScale, ray.Direction.Z / directionScale};
        const float a = Dot(scaledDirection, scaledDirection);
        const float halfB = Dot(offset, scaledDirection);
        const float c = Dot(offset, offset) - radius * radius;
        const float discriminant = halfB * halfB - a * c;
        if (discriminant < 0.0f) return false;
        const float root = sqrtf(discriminant);
        const float q = -halfB - copysignf(root, halfB);
        float first = q == 0.0f ? -halfB / a : q / a;
        float second = q == 0.0f ? first : c / q;
        if (first > second) { const float temporary = first; first = second; second = temporary; }
        float distance = first / directionScale;
        if (distance <= ray.MinimumDistance) distance = second / directionScale;
        if (!isfinite(distance) || distance <= ray.MinimumDistance || distance >= ray.MaximumDistance || distance > hit.Distance)
        {
            return false;
        }
        const Vec3 position = Add(ray.Origin, Multiply(ray.Direction, distance));
        const Vec3 geometricNormal = Normalize(Subtract(position, center));
        Vec3 normal = geometricNormal;
        if (Dot(normal, ray.Direction) > 0.0f) normal = Multiply(normal, -1.0f);
        hit.Distance = distance;
        hit.Normal = normal;
        hit.GeometricNormal = geometricNormal;
        hit.FrontFace = Dot(geometricNormal, scaledDirection) < 0.0f;
        hit.Color = {1.0f, 1.0f, 1.0f};
        hit.MaterialIndex = sphere.MaterialIndex;
        hit.TexCoordX = 0.0f;
        hit.TexCoordY = 0.0f;
        hit.HasTexCoord = 0u;
#if defined(ARVO_SPECTRAL_TRANSPORT)
        hit.VertexSpectrumBase = InvalidPrimitiveIndex;
        hit.Barycentric = make_float2(0, 0);
#endif
        return true;
    }
