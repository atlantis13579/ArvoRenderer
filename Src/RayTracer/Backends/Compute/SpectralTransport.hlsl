// Native compute radiometry. One sampled wavelength is carried by the path and
// every direct connection; only the final film sample divides by its density.
// RGB and spectral branches share geometry and directional BSDF measures, while
// all color products happen after wavelength-space evaluation.
StructuredBuffer<RgbSpectrum> SpectralCoefficients : register(t11);
StructuredBuffer<uint> SpectralReferences : register(t12);
static const uint SpectralTransportBit = 1u << 5;
static const uint WavelengthSampleDimension = 0x57415645u;
static const uint SpectralMaterialBase = 0, SpectralLightBase = 1;
static const uint SpectralTextureBase = 2, SpectralTexelBase = 3;
static const uint SpectralBackgroundTop = 4, SpectralBackgroundBottom = 5;
static const uint SpectralAbsorption = 6, SpectralScattering = 7;
static const uint SpectralMaterialStride = 6, SpectralTextureStride = 4;
static const uint MaterialAlbedoLink = 0, MaterialAmbientLink = 1, MaterialSpecularLink = 2;
static const uint MaterialEmissionLink = 3, MaterialIndexLink = 4, MaterialExtinctionLink = 5;

WavelengthPacket BeginSpectralPath(SampleContext samples)
{
    WavelengthPacket packet = (WavelengthPacket)0;
    if ((TransportFlags & SpectralTransportBit) != 0)
        packet = TerminateSecondaryWavelengths(SampleWavelengths(
            Sample2D(samples, WavelengthSampleDimension).x, (TransportFlags & ImportanceBit) != 0));
    return packet;
}
float SpectralValue(uint entry, uint role, float wavelength)
{
    return EvaluateSceneSpectrum(SpectralCoefficients[entry], role, wavelength);
}
PackedMaterial SpectralMaterial(PackedMaterial material, uint index, float wavelength)
{
    if (wavelength == 0)
        return material;
    const uint first = SpectralReferences[SpectralMaterialBase] + index * SpectralMaterialStride;
    material.AlbedoReflectivity.xyz =
        SpectralValue(SpectralReferences[first + MaterialAlbedoLink], SpectrumRoleReflectance, wavelength)
            .xxx;
    material.AmbientShininess.xyz =
        SpectralValue(SpectralReferences[first + MaterialAmbientLink], SpectrumRoleReflectance, wavelength)
            .xxx;
    material.Specular.xyz =
        SpectralValue(SpectralReferences[first + MaterialSpecularLink], SpectrumRoleReflectance, wavelength)
            .xxx;
    material.Emission.xyz =
        SpectralValue(SpectralReferences[first + MaterialEmissionLink], SpectrumRoleIlluminant, wavelength)
            .xxx;
    material.ConductorIndex.xyz =
        SpectralValue(SpectralReferences[first + MaterialIndexLink], SpectrumRoleNonnegative, wavelength).xxx;
    material.ConductorExtinction.xyz =
        SpectralValue(SpectralReferences[first + MaterialExtinctionLink], SpectrumRoleNonnegative, wavelength)
            .xxx;
    return material;
}
float3 SpectralSurface(HitData hit, uint textureIndex, float wavelength)
{
    if (wavelength == 0)
        return hit.Color * SampleDiffuse(textureIndex, hit.TexCoord, hit.HasTexCoord);
    float value = 1;
    if (hit.VertexSpectrumBase != InvalidPrimitiveIndex)
        value = InterpolateSpectralVertex(
            SpectralCoefficients[SpectralReferences[hit.VertexSpectrumBase]],
            SpectralCoefficients[SpectralReferences[hit.VertexSpectrumBase + 1]],
            SpectralCoefficients[SpectralReferences[hit.VertexSpectrumBase + 2]],
            float3(1 - hit.Barycentric.x - hit.Barycentric.y, hit.Barycentric), wavelength);
    if (hit.HasTexCoord != 0 && textureIndex < TextureCount)
    {
        const uint first = SpectralReferences[SpectralTextureBase] + textureIndex * SpectralTextureStride;
        SpectralTextureDescriptor descriptor;
        descriptor.FirstTexel = SpectralReferences[SpectralTexelBase] + SpectralReferences[first];
        descriptor.Width = SpectralReferences[first + 1];
        descriptor.Height = SpectralReferences[first + 2];
        descriptor.Reserved = 0;
        value *= SampleSpectralTexture(descriptor, hit.TexCoord, wavelength, SpectralReferences,
                                       SpectralCoefficients);
    }
    return value.xxx;
}
float3 SpectralLight(float3 rgb, uint light, float wavelength)
{
    if (wavelength == 0)
        return rgb;
    return SpectralValue(SpectralReferences[SpectralReferences[SpectralLightBase] + light],
                         SpectrumRoleIlluminant, wavelength)
        .xxx;
}
float3 SpectralEnvironment(float3 direction, float wavelength)
{
    const float blend = saturate(direction.y * 0.5f + 0.5f);
    if (wavelength == 0)
        return lerp(BackgroundBottom, BackgroundTop, blend);
    const float lower =
        SpectralValue(SpectralReferences[SpectralBackgroundBottom], SpectrumRoleIlluminant, wavelength);
    const float upper =
        SpectralValue(SpectralReferences[SpectralBackgroundTop], SpectrumRoleIlluminant, wavelength);
    return lerp(lower, upper, blend).xxx;
}
MediumParameters SpectralMedium(MediumParameters medium, float wavelength)
{
    if (wavelength != 0 && medium.Enabled != 0)
    {
        medium.Absorption =
            SpectralValue(SpectralReferences[SpectralAbsorption], SpectrumRoleNonnegative, wavelength).xxx;
        medium.Scattering =
            SpectralValue(SpectralReferences[SpectralScattering], SpectrumRoleNonnegative, wavelength).xxx;
    }
    return medium;
}
float3 SpectralFilm(WavelengthPacket wavelengths, float3 radiance)
{
    if (wavelengths.Lambda[0] == 0)
        return radiance;
    SampledSpectrum spectrum = (SampledSpectrum)0;
    spectrum.Value[0] = radiance.x;
    return XYZToLinearSRGB(SpectrumToXYZ(wavelengths, spectrum));
}
