// CUDA camera paths retain one wavelength and its original sampling density.
// BSDF and medium routines receive replicated monochromatic coefficients;
// radiance is converted once, at the film boundary (PBRT v4, spectral transport).
constexpr unsigned SpectralTransportBit = 1u << 5;
constexpr unsigned WavelengthSampleDimension = 0x57415645u;
constexpr unsigned MaterialLinks = 0, LightLinks = 1, TextureLinks = 2, TexelLinks = 3;
constexpr unsigned SkyLink = 4, GroundLink = 5, AbsorptionLink = 6, ScatteringLink = 7;
constexpr unsigned MaterialLinkStride = 6, TextureLinkStride = 4;
constexpr unsigned AlbedoSlot = 0, AmbientSlot = 1, SpecularSlot = 2;
constexpr unsigned EmissionSlot = 3, ConductorIndexSlot = 4, ConductorExtinctionSlot = 5;
struct PathSpectra
{
    const RgbSpectrum* Palette;
    const unsigned* Links;
    float Wavelength;
};
__device__ WavelengthPacket PathWavelength(const SampleContext& context, unsigned flags)
{
    WavelengthPacket packet{};
    if (flags & SpectralTransportBit)
        packet = TerminateSecondaryWavelengths(
            SampleWavelengths(Sample2D(context, WavelengthSampleDimension).x, (flags & ImportanceBit) != 0));
    return packet;
}
__device__ float PathSpectrum(const PathSpectra& path, unsigned entry, unsigned role)
{
    return EvaluateSceneSpectrum(path.Palette[entry], role, path.Wavelength);
}
__device__ Vec3 SpectralScalar(float value)
{
    return {value, value, value};
}
__device__ void SpectralMaterialColor(Vec4& destination, const PathSpectra& path, unsigned entry,
                                      unsigned role)
{
    const float value = PathSpectrum(path, entry, role);
    destination.X = value;
    destination.Y = value;
    destination.Z = value;
}
__device__ PackedMaterial PathMaterial(PackedMaterial material, unsigned index, const PathSpectra& path)
{
    if (path.Wavelength == 0)
        return material;
    const unsigned* fields = path.Links + path.Links[MaterialLinks] + index * MaterialLinkStride;
    SpectralMaterialColor(material.AlbedoReflectivity, path, fields[AlbedoSlot], SpectrumRoleReflectance);
    SpectralMaterialColor(material.AmbientShininess, path, fields[AmbientSlot], SpectrumRoleReflectance);
    SpectralMaterialColor(material.Specular, path, fields[SpecularSlot], SpectrumRoleReflectance);
    SpectralMaterialColor(material.Emission, path, fields[EmissionSlot], SpectrumRoleIlluminant);
    SpectralMaterialColor(material.ConductorIndex, path, fields[ConductorIndexSlot], SpectrumRoleNonnegative);
    SpectralMaterialColor(material.ConductorExtinction, path, fields[ConductorExtinctionSlot],
                          SpectrumRoleNonnegative);
    return material;
}
__device__ Vec3 PathSurface(const Hit& hit, unsigned textureIndex, const PackedTexture* textures,
                            const unsigned* texels, const FrameConstants& frame, const PathSpectra& path)
{
    if (path.Wavelength == 0)
    {
        const Vec3 texture = hit.HasTexCoord ? SampleDiffuseTexture(textures, texels, frame, textureIndex,
                                                                    hit.TexCoordX, hit.TexCoordY)
                                             : Vec3{1, 1, 1};
        return Multiply(hit.Color, texture);
    }
    float value = 1;
    if (hit.VertexSpectrumBase != InvalidPrimitiveIndex)
        value = InterpolateSpectralVertex(
            path.Palette[path.Links[hit.VertexSpectrumBase]],
            path.Palette[path.Links[hit.VertexSpectrumBase + 1]],
            path.Palette[path.Links[hit.VertexSpectrumBase + 2]],
            make_float3(1 - hit.Barycentric.x - hit.Barycentric.y, hit.Barycentric.x, hit.Barycentric.y),
            path.Wavelength);
    if (hit.HasTexCoord && textureIndex < frame.TextureCount)
    {
        const unsigned* fields = path.Links + path.Links[TextureLinks] + textureIndex * TextureLinkStride;
        SpectralTextureDescriptor texture{path.Links[TexelLinks] + fields[0], fields[1], fields[2], 0};
        value *= SampleSpectralTexture(texture, make_float2(hit.TexCoordX, hit.TexCoordY), path.Wavelength,
                                       path.Links, path.Palette);
    }
    return SpectralScalar(value);
}
__device__ Vec3 PathLight(Vec3 rgb, unsigned index, const PathSpectra& path)
{
    if (path.Wavelength == 0)
        return rgb;
    return SpectralScalar(
        PathSpectrum(path, path.Links[path.Links[LightLinks] + index], SpectrumRoleIlluminant));
}
__device__ Vec3 PathEnvironment(const FrameConstants& frame, Vec3 direction, const PathSpectra& path)
{
    if (path.Wavelength == 0)
        return Background(frame, direction);
    const float blend = fminf(1, fmaxf(0, direction.Y * 0.5f + 0.5f));
    const float lower = PathSpectrum(path, path.Links[GroundLink], SpectrumRoleIlluminant);
    const float upper = PathSpectrum(path, path.Links[SkyLink], SpectrumRoleIlluminant);
    return SpectralScalar(lower * (1 - blend) + upper * blend);
}
__device__ MediumParameters PathMedium(MediumParameters source, const PathSpectra& path)
{
    if (source.Enabled && path.Wavelength != 0)
    {
        const float absorption = PathSpectrum(path, path.Links[AbsorptionLink], SpectrumRoleNonnegative);
        const float scattering = PathSpectrum(path, path.Links[ScatteringLink], SpectrumRoleNonnegative);
        source.Absorption = make_float3(absorption, absorption, absorption);
        source.Scattering = make_float3(scattering, scattering, scattering);
    }
    return source;
}
__device__ Vec3 PathFilm(Vec3 radiance, const WavelengthPacket& wavelengths)
{
    if (wavelengths.Lambda[0] == 0)
        return radiance;
    SampledSpectrum spectrum{};
    spectrum.Value[0] = radiance.X;
    const float3 color = XYZToLinearSRGB(SpectrumToXYZ(wavelengths, spectrum));
    return {color.x, color.y, color.z};
}
