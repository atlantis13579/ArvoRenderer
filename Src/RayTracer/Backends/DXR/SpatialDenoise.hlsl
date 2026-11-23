// Independent DXR offline reconstruction; preview averages stationary linear frames.
// Dammertz et al. (2010), https://doi.org/10.2312/EGGH/HPG10/067-075.
// A ray-generation dispatch owns image work; filtering launches no scene rays.
// Optional static pinhole guides reject normal/plane/material discontinuities.
// They do not represent finite-aperture, exposure or volume observations.
struct RawPixel
{
    float3 Radiance;
    float LuminanceM2;
    uint Samples;
    uint GuideFlags, GuideMaterial, Padding;
    float3 GuidePosition;
    float GuideDepth;
    float3 GuideNormal;
    float ReservedNormal;
    float3 DiffuseAlbedo;
    float ReservedAlbedo;
};
StructuredBuffer<RawPixel> Film : register(t0);
StructuredBuffer<float4> Prior : register(t1);
RWStructuredBuffer<float4> Filtered : register(u0);
RWStructuredBuffer<uint> Failure : register(u1);
RWStructuredBuffer<uint> Display : register(u2);
cbuffer Reconstruction : register(b0)
{
    uint FilmWidth, FilmHeight, Level, First;
    uint Population, Features;
    float NoiseWidth, RelativeWidth;
    float AbsoluteWidth;
    uint EncodeGamma;
    float NormalWidth, RelativePlaneWidth;
};
static const uint MaximumLevels = 5, UseDispersion = 1, UseGeometry = 2;
static const uint BadDispatch = 1, BadValue = 2;
static const int KernelRadius = 2;
static const double RedLuminance = 0.2126, GreenLuminance = 0.7152, BlueLuminance = 0.0722;
double3 Source(uint address)
{
    if (Level == 0)
        return double3(Film[address].Radiance);
    return double3(Prior[address].xyz);
}
double PositiveRoot(double value)
{
    if (value == 0) return 0;
    // Normalize the exponent before the native binary32 seed, then correct in
    // double arithmetic. All values derived from finite film moments are normal.
    static const int DoubleBias = 1023;
    static const uint MantissaHighBits = 20, MantissaHighMask = 0xfffff, ExponentMask = 0x7ff;
    static const uint Refinements = 2;
    uint low, high; asuint(value, low, high);
    const int exponent = int((high >> MantissaHighBits) & ExponentMask) - DoubleBias;
    double mantissa = asdouble(low, (high & MantissaHighMask) | (uint(DoubleBias) << MantissaHighBits));
    if (exponent & 1) mantissa *= 2;
    const double scale = asdouble(0u, uint((exponent >> 1) + DoubleBias) << MantissaHighBits);
    double root = double(sqrt(float(mantissa))) * scale;
    for (uint correction = 0; correction < Refinements; ++correction) root = (root + value / root) / 2;
    return root;
}
double Deviation(uint address)
{
    const RawPixel pixel = Film[address];
    if (!(Features & UseDispersion) || pixel.Samples <= 1)
        return 0;
    return PositiveRoot(double(pixel.LuminanceM2) / double(pixel.Samples) / double(pixel.Samples - 1));
}
double Kernel(int distance)
{
    if (distance == 0)
        return 6;
    if (abs(distance) == 1)
        return 4;
    return 1;
}
double Similarity(double3 a, double3 b, double deviationA, double deviationB)
{
    const double maximum = max(double(AbsoluteWidth),
        max(max(max(abs(a.x), abs(a.y)), abs(a.z)), max(max(abs(b.x), abs(b.y)), abs(b.z))));
    a /= maximum;
    b /= maximum;
    const double ya = a.x * RedLuminance + a.y * GreenLuminance + a.z * BlueLuminance;
    const double yb = b.x * RedLuminance + b.y * GreenLuminance + b.z * BlueLuminance;
    const double na = deviationA / maximum, nb = deviationB / maximum;
    const double sigma =
        AbsoluteWidth / maximum + RelativeWidth * max(abs(ya), abs(yb)) + NoiseWidth * PositiveRoot(na * na + nb * nb);
    const double3 difference = a - b;
    const double square =
        difference.x * difference.x + difference.y * difference.y + difference.z * difference.z;
    return double(exp(float(-square / (2 * sigma * sigma))));
}
double SurfaceAgreement(uint first, uint second)
{
    if ((Features & UseGeometry) == 0) return 1;
    const RawPixel a = Film[first], b = Film[second];
    static const uint SurfaceBit = 1;
    const bool surfaceA = (a.GuideFlags & SurfaceBit) != 0;
    const bool surfaceB = (b.GuideFlags & SurfaceBit) != 0;
    if (surfaceA != surfaceB) return 0;
    if (!surfaceA) return 1;
    if (a.GuideMaterial != b.GuideMaterial) return 0;
    const double x = double(b.GuidePosition.x) - a.GuidePosition.x;
    const double y = double(b.GuidePosition.y) - a.GuidePosition.y;
    const double z = double(b.GuidePosition.z) - a.GuidePosition.z;
    const double planeA = abs(x * a.GuideNormal.x + y * a.GuideNormal.y + z * a.GuideNormal.z);
    const double planeB = abs(x * b.GuideNormal.x + y * b.GuideNormal.y + z * b.GuideNormal.z);
    const double planeSigma = double(RelativePlaneWidth) * max(a.GuideDepth, b.GuideDepth);
    const double planeError = max(planeA, planeB) / planeSigma;
    const double nx = double(a.GuideNormal.x) - b.GuideNormal.x;
    const double ny = double(a.GuideNormal.y) - b.GuideNormal.y;
    const double nz = double(a.GuideNormal.z) - b.GuideNormal.z;
    const double normalError = (nx * nx + ny * ny + nz * nz) / (double(NormalWidth) * NormalWidth);
    return double(exp(float(-(normalError + planeError * planeError) / 2)));
}
uint PackDisplay(float3 rgb)
{
    static const float SrgbThreshold = 0.0031308, SrgbSlope = 12.92, SrgbScale = 1.055;
    static const float SrgbOffset = 0.055, SrgbExponent = 1.0 / 2.4, ByteScale = 255;
    static const uint ChannelBits = 8, AlphaMask = 0xff000000u;
    rgb /= max(1.0, max(rgb.x, max(rgb.y, rgb.z)));
    uint packed = AlphaMask;
    for (uint channel = 0; channel < 3; ++channel)
    {
        float value = saturate(rgb[channel]);
        if (EncodeGamma)
            value = value <= SrgbThreshold ? SrgbSlope * value
                                           : SrgbScale * pow(value, SrgbExponent) - SrgbOffset;
        packed |= uint(value * ByteScale + 0.5) << (channel * ChannelBits);
    }
    return packed;
}
[shader("raygeneration")]
void SpatialDenoiseRayGen() {
    const uint lane = DispatchRaysIndex().x;
    if (lane >= Population)
        return;
    // Root SRVs carry an address, not a descriptor extent; the host validates
    // allocation lengths and this shader validates the complete dispatch domain.
    static const uint MaximumExtent = 16384;
    const uint available = FilmWidth * FilmHeight;
    if (!FilmWidth || !FilmHeight || FilmWidth > MaximumExtent || FilmHeight > MaximumExtent ||
        Level >= MaximumLevels || First > available ||
        Population > available - First || (Features & ~(UseDispersion | UseGeometry)) || !(AbsoluteWidth > 0) ||
        ((Features & UseGeometry) && (!(NormalWidth > 0) || !(RelativePlaneWidth > 0))))
    {
        InterlockedOr(Failure[0], BadDispatch);
        return;
    }
    const uint address = First + lane;
    const int x = int(address % FilmWidth), y = int(address / FilmWidth), stridePixels = int(1u << Level);
    const double3 center = Source(address);
    const double noise = Deviation(address);
    double3 average = 0;
    const bool diffuse = Film[address].DiffuseAlbedo.x >= 0;
    double3 albedoSum = 0;
    double normalization = 0;
    for (int vertical = -KernelRadius; vertical <= KernelRadius; ++vertical)
        for (int horizontal = -KernelRadius; horizontal <= KernelRadius; ++horizontal)
        {
            const int sx = x + horizontal * stridePixels, sy = y + vertical * stridePixels;
            if (sx < 0 || sy < 0 || sx >= int(FilmWidth) || sy >= int(FilmHeight))
                continue;
            const uint location = uint(sy) * FilmWidth + uint(sx);
            if (diffuse != (Film[location].DiffuseAlbedo.x >= 0))
                continue;
            const double3 neighbor = Source(location);
            const double weight = Kernel(vertical) * Kernel(horizontal) *
                                  Similarity(center, neighbor, noise, Deviation(location)) *
                                  SurfaceAgreement(address, location);
            if (!(weight > 0))
                continue;
            if (diffuse)
                albedoSum += weight * double3(Film[location].DiffuseAlbedo);
            const double preceding = normalization;
            normalization += weight;
            // Keep the small weight directly; 1 - a nearly unit blend loses
            // dark-tail radiance through cancellation.
            average = weight > preceding
                ? neighbor + (average - neighbor) * (preceding / normalization)
                : average + (neighbor - average) * (weight / normalization);
        }
    if (diffuse && normalization > 0)
    {
        static const double AlbedoFloor = 0.001;
        const double3 ratio = (double3(Film[address].DiffuseAlbedo) + AlbedoFloor) /
                              (albedoSum / normalization + AlbedoFloor);
        const double3 adjusted = average * ratio;
        for (uint channel = 0; channel < 3; ++channel)
            if (isfinite(float(adjusted[channel])))
                average[channel] = adjusted[channel];
    }
    const float3 radiance = float3(average);
    if (normalization <= 0 || any(!isfinite(radiance)))
    {
        InterlockedOr(Failure[0], BadValue);
        return;
    }
    Filtered[address] = float4(radiance, 0);
    Display[address] = PackDisplay(radiance);
}
