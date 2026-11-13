// PBRT v4, "BSDF Representation": the shading frame defines f and its PDFs;
// the geometric normal selects media and rejects inconsistent hemispheres.
// https://pbr-book.org/4ed/Reflection_Models/BSDF_Representation
// This CUDA adapter composes only this backend's scalar scattering kernels.
// Textures tint diffuse reflectance; conductor optical constants and dielectric
// transmission remain independent of the legacy diffuse color.
struct PhysicalSurface
{
    Vec3 Normal, Geometry, Tangent, Bitangent, Reflectance;
    float3 View;
    float IncidentIndex, TransmittedIndex, ViewGeometryCosine;
    unsigned FrontFace, Mode;
    PackedMaterial Material;
};
struct PhysicalEvaluation
{
    Vec3 Value;
    float Pdf, ReversePdf;
};
struct PhysicalSample
{
    Vec3 Direction, Weight;
    float Pdf, ReversePdf, RelativeIndex;
    unsigned Event;
};

__device__ float3 SurfaceLocal(const PhysicalSurface& surface, Vec3 direction)
{
    return make_float3(Dot(direction, surface.Tangent), Dot(direction, surface.Bitangent),
                       Dot(direction, surface.Normal));
}
__device__ Vec3 SurfaceWorld(const PhysicalSurface& surface, float3 direction)
{
    return Normalize(
        Add(Add(Multiply(surface.Tangent, direction.x), Multiply(surface.Bitangent, direction.y)),
            Multiply(surface.Normal, direction.z)));
}
__device__ PhysicalSurface MakePhysicalSurface(const PackedMaterial& material, const Hit& hit, Vec3 view,
                                               Vec3 color, unsigned mode = ArvoCuda::TransportRadiance)
{
    PhysicalSurface surface{};
    surface.Material = material;
    surface.Normal = hit.Normal;
    surface.Geometry = hit.GeometricNormal;
    surface.FrontFace = hit.FrontFace;
    surface.Mode = mode;
    surface.ViewGeometryCosine = fabsf(Dot(Normalize(view), hit.GeometricNormal));
    const Vec3 helper = fabsf(hit.Normal.Y) < BasisParallelThreshold ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    surface.Tangent = Normalize(Cross(helper, hit.Normal));
    surface.Bitangent = Cross(hit.Normal, surface.Tangent);
    surface.View = SurfaceLocal(surface, Normalize(view));
    const Vec3 reflectance = Multiply(ToVec3(material.AlbedoReflectivity), color);
    surface.Reflectance = {fminf(1, fmaxf(0, reflectance.X)), fminf(1, fmaxf(0, reflectance.Y)),
                           fminf(1, fmaxf(0, reflectance.Z))};
    surface.IncidentIndex = hit.FrontFace ? material.Optical.Y : material.Optical.Z;
    surface.TransmittedIndex = hit.FrontFace ? material.Optical.Z : material.Optical.Y;
    return surface;
}
__device__ bool PhysicalDelta(const PhysicalSurface& surface)
{
    return surface.Material.Model != DiffuseMaterial &&
           (surface.Material.Optical.X == 0 || (surface.Material.Model == DielectricMaterial &&
                                                surface.IncidentIndex == surface.TransmittedIndex));
}
__device__ bool SurfaceHemisphere(const PhysicalSurface& surface, Vec3 direction, bool transmission)
{
    const float cosine = Dot(surface.Geometry, direction);
    return cosine != 0 && (((cosine > 0) == (surface.FrontFace != 0)) != transmission);
}
__device__ Vec3 PhysicalRayOrigin(const PhysicalSurface& surface, Vec3 position, Vec3 direction)
{
    return Add(position,
               Multiply(surface.Geometry, Dot(surface.Geometry, direction) >= 0 ? RayEpsilon : -RayEpsilon));
}
__device__ float SurfaceAdjoint(const PhysicalSurface& surface, Vec3 direction)
{
    // Importance transport needs Veach's shading-normal adjoint (1997, ch. 5).
    // Its cosine ratio changes energy only; forward/reverse PDFs still describe
    // the original proposal, including clipped null events and discrete masses.
    if (surface.Mode == ArvoCuda::TransportRadiance) return 1;
    const float denominator = surface.ViewGeometryCosine * fabsf(Dot(surface.Normal, direction));
    return denominator > 0 ? fabsf(surface.View.z * Dot(surface.Geometry, direction)) / denominator : 0;
}
__device__ PhysicalEvaluation EvaluatePhysical(const PhysicalSurface& surface, Vec3 direction,
                                               bool importance)
{
    const float3 local = SurfaceLocal(surface, direction);
    if (surface.View.z <= 0 || !SurfaceHemisphere(surface, direction, local.z < 0))
        return {};
    const PackedMaterial& material = surface.Material;
    const float2 slope = make_float2(material.Optical.X, material.Optical.X);
    if (material.Model == DiffuseMaterial)
    {
        const auto diffuse = ArvoCuda::EvaluateDiffuse(surface.View, local, 1, importance);
        return {Multiply(surface.Reflectance, diffuse.Value * SurfaceAdjoint(surface, direction)),
                diffuse.Pdf, diffuse.ReversePdf};
    }
    if (material.Model == ConductorMaterial)
    {
        const auto red = ArvoCuda::EvaluateConductor(surface.View, local, slope, material.ConductorIndex.X,
                                                     material.ConductorExtinction.X);
        const auto green = ArvoCuda::EvaluateConductor(surface.View, local, slope, material.ConductorIndex.Y,
                                                       material.ConductorExtinction.Y);
        const auto blue = ArvoCuda::EvaluateConductor(surface.View, local, slope, material.ConductorIndex.Z,
                                                      material.ConductorExtinction.Z);
        const float pdf = !importance && !PhysicalDelta(surface) && local.z > 0 ? 1 / (2 * Pi) : red.Pdf;
        const float reversePdf = !importance && !PhysicalDelta(surface) && local.z > 0
            ? 1 / (2 * Pi) : red.ReversePdf;
        return {Multiply(Vec3{red.Value, green.Value, blue.Value}, SurfaceAdjoint(surface, direction)), pdf, reversePdf};
    }
    if (material.Model == DielectricMaterial)
    {
        const auto glass =
            ArvoCuda::EvaluateDielectric(surface.View, local, slope, surface.IncidentIndex,
                                         surface.TransmittedIndex, surface.Mode);
        const float pdf = !importance && !PhysicalDelta(surface) && local.z != 0 ? 1 / (4 * Pi) : glass.Pdf;
        const float reversePdf = !importance && !PhysicalDelta(surface) && local.z != 0
            ? 1 / (4 * Pi) : glass.ReversePdf;
        const float value = glass.Value * SurfaceAdjoint(surface, direction);
        return {{value, value, value}, pdf, reversePdf};
    }
    return {};
}
__device__ PhysicalSample SamplePhysical(const PhysicalSurface& surface, float2 variates, float lobe,
                                         bool importance)
{
    if (surface.View.z <= 0)
        return {};
    const PackedMaterial& material = surface.Material;
    // Uniform proposals keep null samples. Resampling clipped directions would
    // change their normalization and invalidate the matching direct-light PDF.
    if (!importance && !PhysicalDelta(surface) && material.Model != DiffuseMaterial)
    {
        const float cosine = material.Model == DielectricMaterial ? 1 - 2 * variates.x : variates.x;
        // The exact proposal horizon has zero measure; do not let a world-frame
        // roundtrip turn it into a near-zero denominator in the adjoint.
        if (cosine == 0) return {};
        const float sine = sqrtf(fmaxf(0, 1 - cosine * cosine));
        const float azimuth = 2 * Pi * variates.y;
        const Vec3 direction =
            SurfaceWorld(surface, make_float3(sine * cosf(azimuth), sine * sinf(azimuth), cosine));
        const PhysicalEvaluation evaluation = EvaluatePhysical(surface, direction, false);
        if (evaluation.Pdf <= 0)
            return {};
        return {direction, Multiply(evaluation.Value, fabsf(Dot(surface.Normal, direction)) / evaluation.Pdf),
                evaluation.Pdf, evaluation.ReversePdf, cosine < 0 ? surface.TransmittedIndex / surface.IncidentIndex : 1,
                ArvoCuda::ScatteringGlossy |
                    (cosine < 0 ? ArvoCuda::ScatteringTransmission : ArvoCuda::ScatteringReflection)};
    }
    ArvoCuda::BsdfSample scalar{};
    Vec3 value{};
    const float2 slope = make_float2(material.Optical.X, material.Optical.X);
    if (material.Model == DiffuseMaterial)
    {
        scalar = ArvoCuda::SampleDiffuse(surface.View, 1, variates, importance);
        value = Multiply(surface.Reflectance, scalar.Value);
    }
    else if (material.Model == ConductorMaterial)
    {
        // GGX directional sampling is achromatic. Only Fresnel needs RGB evaluation.
        scalar = ArvoCuda::SampleConductor(surface.View, slope, material.ConductorIndex.X,
                                           material.ConductorExtinction.X, variates);
        value.X = scalar.Value;
        if (scalar.Event & ArvoCuda::ScatteringDelta)
        {
            value.Y = ArvoCuda::ConductorFresnel(surface.View.z, material.ConductorIndex.Y,
                                                 material.ConductorExtinction.Y) /
                      surface.View.z;
            value.Z = ArvoCuda::ConductorFresnel(surface.View.z, material.ConductorIndex.Z,
                                                 material.ConductorExtinction.Z) /
                      surface.View.z;
        }
        else if (scalar.Event != 0)
        {
            value.Y = ArvoCuda::EvaluateConductor(surface.View, scalar.Direction, slope,
                                                  material.ConductorIndex.Y, material.ConductorExtinction.Y)
                          .Value;
            value.Z = ArvoCuda::EvaluateConductor(surface.View, scalar.Direction, slope,
                                                  material.ConductorIndex.Z, material.ConductorExtinction.Z)
                          .Value;
        }
    }
    else if (material.Model == DielectricMaterial)
    {
        scalar =
            ArvoCuda::SampleDielectric(surface.View, slope, surface.IncidentIndex, surface.TransmittedIndex,
                                       surface.Mode, variates, lobe);
        value = {scalar.Value, scalar.Value, scalar.Value};
    }
    if (scalar.Event == 0 || scalar.Pdf <= 0)
        return {};
    const Vec3 direction = SurfaceWorld(surface, scalar.Direction);
    if (!SurfaceHemisphere(surface, direction, (scalar.Event & ArvoCuda::ScatteringTransmission) != 0))
        return {};
    return {direction, Multiply(value, fabsf(scalar.Direction.z) / scalar.Pdf * SurfaceAdjoint(surface, direction)),
            scalar.Pdf, scalar.ReversePdf, scalar.RelativeIndex, scalar.Event};
}
