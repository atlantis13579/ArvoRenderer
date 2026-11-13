// MSL spectral camera transport. The hero wavelength is explicit path state,
// including direct connections and free flights. Conversion to signed RGB is a
// film operation; no color product or filter runs on reconstructed coefficients.
constant uint SpectralTransportBit = 1u << 5;
constant uint WavelengthSampleDimension = 0x57415645u;
constant uint SpectralMaterialTable = 0, SpectralLightTable = 1, SpectralTextureTable = 2;
constant uint SpectralTexelTable = 3, SpectralSky = 4, SpectralGround = 5;
constant uint SpectralAbsorption = 6, SpectralScattering = 7;
constant uint SpectralMaterialWords = 6, SpectralTextureWords = 4;
constant uint AlbedoField = 0, AmbientField = 1, SpecularField = 2;
constant uint EmissionField = 3, ConductorIndexField = 4, ConductorExtinctionField = 5;
struct SpectralPathContext
{
    device const RgbSpectrum* Coefficients;
    device const uint* References;
    float Wavelength;
};
WavelengthPacket BeginSpectralPath(SampleContext samples, uint flags)
{
    WavelengthPacket wavelengths{};
    if ((flags & SpectralTransportBit) != 0)
        wavelengths = TerminateSecondaryWavelengths(
            SampleWavelengths(Sample2D(samples, WavelengthSampleDimension).x, (flags & ImportanceBit) != 0));
    return wavelengths;
}
float SpectralCoefficient(SpectralPathContext context, uint entry, uint role)
{
    return EvaluateSceneSpectrum(context.Coefficients[entry], role, context.Wavelength);
}
PackedMaterial SpectralMaterial(SpectralPathContext context, PackedMaterial material, uint index)
{
    if (context.Wavelength == 0)
        return material;
    device const uint* fields =
        context.References + context.References[SpectralMaterialTable] + index * SpectralMaterialWords;
    material.AlbedoReflectivity.xyz =
        float3(SpectralCoefficient(context, fields[AlbedoField], SpectrumRoleReflectance));
    material.AmbientShininess.xyz =
        float3(SpectralCoefficient(context, fields[AmbientField], SpectrumRoleReflectance));
    material.Specular.xyz =
        float3(SpectralCoefficient(context, fields[SpecularField], SpectrumRoleReflectance));
    material.Emission.xyz =
        float3(SpectralCoefficient(context, fields[EmissionField], SpectrumRoleIlluminant));
    material.ConductorIndex.xyz =
        float3(SpectralCoefficient(context, fields[ConductorIndexField], SpectrumRoleNonnegative));
    material.ConductorExtinction.xyz =
        float3(SpectralCoefficient(context, fields[ConductorExtinctionField], SpectrumRoleNonnegative));
    return material;
}
float3 SpectralSurface(SpectralPathContext context, HitData hit, uint textureIndex,
                       device const PackedTextureDescriptor* textures, device const uint* texels,
                       constant FrameConstants& frame)
{
    if (context.Wavelength == 0)
        return max(hit.Color, float3(0)) *
               max(SampleDiffuse(textures, texels, frame, textureIndex, hit.TexCoord, hit.HasTexCoord),
                   float3(0));
    float color = 1;
    if (hit.VertexSpectrumBase != InvalidPrimitiveIndex)
    {
        const float3 weights = float3(1 - hit.Barycentric.x - hit.Barycentric.y,
                                     hit.Barycentric.x, hit.Barycentric.y);
        color =
            InterpolateSpectralVertex(context.Coefficients[context.References[hit.VertexSpectrumBase]],
                                      context.Coefficients[context.References[hit.VertexSpectrumBase + 1]],
                                      context.Coefficients[context.References[hit.VertexSpectrumBase + 2]],
                                      weights, context.Wavelength);
    }
    if (hit.HasTexCoord != 0 && textureIndex < frame.TextureCount)
    {
        device const uint* fields = context.References + context.References[SpectralTextureTable] +
                                    textureIndex * SpectralTextureWords;
        SpectralTextureDescriptor texture{context.References[SpectralTexelTable] + fields[0], fields[1],
                                          fields[2], 0};
        color *= SampleSpectralTexture(texture, hit.TexCoord, context.Wavelength, context.References,
                                       context.Coefficients);
    }
    return float3(color);
}
float3 SpectralLight(SpectralPathContext context, float3 rgb, uint index)
{
    if (context.Wavelength == 0)
        return rgb;
    uint entry = context.References[context.References[SpectralLightTable] + index];
    return float3(SpectralCoefficient(context, entry, SpectrumRoleIlluminant));
}
float3 SpectralBackground(SpectralPathContext context, constant FrameConstants& frame, float3 direction)
{
    if (context.Wavelength == 0)
        return Background(frame, direction);
    const float blend = clamp(direction.y * 0.5f + 0.5f, 0.0f, 1.0f);
    const float bottom =
        SpectralCoefficient(context, context.References[SpectralGround], SpectrumRoleIlluminant);
    const float top = SpectralCoefficient(context, context.References[SpectralSky], SpectrumRoleIlluminant);
    return float3(bottom * (1 - blend) + top * blend);
}
MediumParameters SpectralMedium(SpectralPathContext context, MediumParameters medium)
{
    if (context.Wavelength != 0 && medium.Enabled != 0)
    {
        medium.Absorption = packed_float3(float3(
            SpectralCoefficient(context, context.References[SpectralAbsorption], SpectrumRoleNonnegative)));
        medium.Scattering = packed_float3(float3(
            SpectralCoefficient(context, context.References[SpectralScattering], SpectrumRoleNonnegative)));
    }
    return medium;
}
float3 SpectralFilm(WavelengthPacket packet, float3 radiance)
{
    if (packet.Lambda[0] == 0)
        return radiance;
    SampledSpectrum value{};
    value.Value[0] = radiance.x;
    return XYZToLinearSRGB(SpectrumToXYZ(packet, value));
}
