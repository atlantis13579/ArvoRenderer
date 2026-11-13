// Offline DXR spectral estimator; preview selects RGB. A hero wavelength
// survives every surface/medium event and light connection. RGB conversion is
// deferred to the film, retaining signed color-matching contributions.
StructuredBuffer<RgbSpectrum> SpectralPalette : register(t14);
StructuredBuffer<uint> SpectralLinks : register(t15);
static const uint SpectralTransportBit = 1u << 5;
static const uint WavelengthSampleDimension = 0x57415645u;
static const uint SpectrumMaterials = 0, SpectrumLights = 1, SpectrumTextures = 2, SpectrumTexels = 3;
static const uint SpectrumSky = 4, SpectrumGround = 5, SpectrumAbsorption = 6, SpectrumScattering = 7;
static const uint MaterialSpectrumStride = 6, TextureSpectrumStride = 4;
static const uint AlbedoSpectrum = 0, AmbientSpectrum = 1, SpecularSpectrum = 2;
static const uint EmissionSpectrum = 3, ConductorIndexSpectrum = 4, ExtinctionSpectrum = 5;

WavelengthPacket SamplePathWavelength(SampleContext context)
{
    WavelengthPacket wavelengths = (WavelengthPacket)0;
    if ((TransportFlags & SpectralTransportBit) != 0)
    {
        const float u = Sample2D(context, WavelengthSampleDimension).x;
        wavelengths =
            TerminateSecondaryWavelengths(SampleWavelengths(u, (TransportFlags & ImportanceBit) != 0));
    }
    return wavelengths;
}
float EvaluatePathSpectrum(uint reference, uint role, float wavelength)
{
    return EvaluateSceneSpectrum(SpectralPalette[reference], role, wavelength);
}
MaterialData EvaluateSpectralMaterial(MaterialData source, uint materialIndex, float wavelength)
{
    if (wavelength == 0)
        return source;
    uint base = SpectralLinks[SpectrumMaterials] + MaterialSpectrumStride * materialIndex;
    source.AlbedoReflectivity.xyz =
        EvaluatePathSpectrum(SpectralLinks[base + AlbedoSpectrum], SpectrumRoleReflectance, wavelength).xxx;
    source.AmbientShininess.xyz =
        EvaluatePathSpectrum(SpectralLinks[base + AmbientSpectrum], SpectrumRoleReflectance, wavelength).xxx;
    source.Specular.xyz =
        EvaluatePathSpectrum(SpectralLinks[base + SpecularSpectrum], SpectrumRoleReflectance, wavelength).xxx;
    source.Emission.xyz =
        EvaluatePathSpectrum(SpectralLinks[base + EmissionSpectrum], SpectrumRoleIlluminant, wavelength).xxx;
    source.ConductorIndex.xyz = EvaluatePathSpectrum(SpectralLinks[base + ConductorIndexSpectrum],
                                                     SpectrumRoleNonnegative, wavelength)
                                    .xxx;
    source.ConductorExtinction.xyz =
        EvaluatePathSpectrum(SpectralLinks[base + ExtinctionSpectrum], SpectrumRoleNonnegative, wavelength)
            .xxx;
    return source;
}
float3 EvaluateSpectralSurface(Payload hit, uint textureIndex, float wavelength)
{
    if (wavelength == 0)
        return max(hit.Color, 0.0f) *
               max(SampleDiffuse(textureIndex, hit.TexCoord, hit.Flags & SurfaceTexCoordFlag), 0.0f);
    float vertex = 1;
    if (hit.VertexSpectrumBase != InvalidPrimitiveIndex)
    {
        float3 weights = float3(1 - hit.Barycentric.x - hit.Barycentric.y, hit.Barycentric);
        vertex = InterpolateSpectralVertex(SpectralPalette[SpectralLinks[hit.VertexSpectrumBase]],
                                           SpectralPalette[SpectralLinks[hit.VertexSpectrumBase + 1]],
                                           SpectralPalette[SpectralLinks[hit.VertexSpectrumBase + 2]],
                                           weights, wavelength);
    }
    if ((hit.Flags & SurfaceTexCoordFlag) != 0 && textureIndex < TextureCount)
    {
        uint location = SpectralLinks[SpectrumTextures] + textureIndex * TextureSpectrumStride;
        SpectralTextureDescriptor descriptor;
        descriptor.FirstTexel = SpectralLinks[SpectrumTexels] + SpectralLinks[location];
        descriptor.Width = SpectralLinks[location + 1];
        descriptor.Height = SpectralLinks[location + 2];
        descriptor.Reserved = 0;
        vertex *= SampleSpectralTexture(descriptor, hit.TexCoord, wavelength, SpectralLinks, SpectralPalette);
    }
    return vertex.xxx;
}
float3 EvaluateSpectralLight(float3 color, uint lightIndex, float wavelength)
{
    if (wavelength == 0)
        return color;
    const uint reference = SpectralLinks[SpectralLinks[SpectrumLights] + lightIndex];
    return EvaluatePathSpectrum(reference, SpectrumRoleIlluminant, wavelength).xxx;
}
float3 EvaluateSpectralBackground(float3 direction, float wavelength)
{
    const float fraction = saturate(direction.y * 0.5f + 0.5f);
    if (wavelength == 0)
        return Background(direction);
    const float sky = EvaluatePathSpectrum(SpectralLinks[SpectrumSky], SpectrumRoleIlluminant, wavelength);
    const float ground =
        EvaluatePathSpectrum(SpectralLinks[SpectrumGround], SpectrumRoleIlluminant, wavelength);
    return (ground * (1 - fraction) + sky * fraction).xxx;
}
MediumParameters EvaluateSpectralMedium(MediumParameters medium, float wavelength)
{
    if (medium.Enabled != 0 && wavelength != 0)
    {
        medium.Absorption =
            EvaluatePathSpectrum(SpectralLinks[SpectrumAbsorption], SpectrumRoleNonnegative, wavelength).xxx;
        medium.Scattering =
            EvaluatePathSpectrum(SpectralLinks[SpectrumScattering], SpectrumRoleNonnegative, wavelength).xxx;
    }
    return medium;
}
float3 SpectralFilmSample(float3 radiance, WavelengthPacket wavelengths)
{
    if (wavelengths.Lambda[0] == 0)
        return radiance;
    SampledSpectrum value = (SampledSpectrum)0;
    value.Value[0] = radiance.x;
    return XYZToLinearSRGB(SpectrumToXYZ(wavelengths, value));
}
