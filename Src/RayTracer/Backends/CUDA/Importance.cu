// CUDA native Importance sampling. No other backend implementation is included.
// Lambertian importance sampling uses p=cos(theta)/pi; uniform hemisphere sampling uses p=1/(2*pi).
    __device__ float DiffusePdf(float cosine, unsigned int flags)
    {
        if (cosine <= 0.0f) return 0.0f;
        return (flags & ImportanceBit) != 0u ? cosine / Pi : 1.0f / (2.0f * Pi);
    }

    __device__ Vec3 SampleDiffuseDirection(Vec3 normal, float2 variates, unsigned int flags)
    {
        const float u = variates.x;
        const bool importance = (flags & ImportanceBit) != 0u;
        const float uniformCosine = 1.0f - u;
        const float radius = importance ? sqrtf(u) : sqrtf(fmaxf(0.0f, 1.0f - uniformCosine * uniformCosine));
        const float cosine = importance ? sqrtf(fmaxf(0.0f, 1.0f - radius * radius)) : uniformCosine;
        const float phi = 2.0f * Pi * variates.y;
        const Vec3 helper = fabsf(normal.Y) < BasisParallelThreshold ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
        const Vec3 tangent = Normalize(Cross(helper, normal));
        const Vec3 bitangent = Cross(normal, tangent);
        return Normalize(Add(Add(Multiply(tangent, radius * cosf(phi)),
            Multiply(bitangent, radius * sinf(phi))), Multiply(normal, cosine)));
    }
