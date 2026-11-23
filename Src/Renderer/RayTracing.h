#pragma once

#include "Renderer/RenderTechniques.h"
#include "Renderer/MotionData.h"
#include "Renderer/MicropolygonOptions.h"

#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32) && !defined(ARVO_RENDERER_STATIC)
#  if defined(ARVO_RENDERER_EXPORTS)
#    define ARVO_RENDERER_API __declspec(dllexport)
#  else
#    define ARVO_RENDERER_API __declspec(dllimport)
#  endif
#else
#  define ARVO_RENDERER_API
#endif

namespace Arvo
{
namespace RayTracing
{
    ARVO_RENDERER_API float Dot(const Float3& lhs, const Float3& rhs);
    ARVO_RENDERER_API Float3 Cross(const Float3& lhs, const Float3& rhs);
    ARVO_RENDERER_API Float3 Normalize(const Float3& value);

    struct Ray
    {
        Float3 Origin;
        Float3 Direction = Float3(0.0f, 0.0f, -1.0f);
        float MinDistance = 1.0e-4f;
        float MaxDistance = 1.0e30f;
    };

    inline constexpr uint32_t InvalidRayTextureIndex =
        std::numeric_limits<uint32_t>::max();

    struct RayTexture
    {
        uint32_t Width = 0;
        uint32_t Height = 0;
        // Immutable RGBA8 texels in top-left row order. Diffuse RGB uses the
        // exact display-sRGB transfer and alpha is linear UNORM. Sampling is
        // bilinear repeat with a V flip; invalid references sample opaque
        // white. Shared ownership lets adapters alias storage without copying.
        std::shared_ptr<const std::vector<uint8_t>> PixelsRGBA8;

        [[nodiscard]] bool IsValid() const noexcept
        {
            if (Width == 0 || Height == 0 || !PixelsRGBA8)
            {
                return false;
            }
            constexpr size_t ComponentsPerPixel = 4U;
            const size_t width = static_cast<size_t>(Width);
            const size_t height = static_cast<size_t>(Height);
            if (height > std::numeric_limits<size_t>::max() / width)
            {
                return false;
            }
            const size_t pixelCount = width * height;
            return pixelCount <=
                       std::numeric_limits<size_t>::max() / ComponentsPerPixel &&
                   PixelsRGBA8->size() == pixelCount * ComponentsPerPixel;
        }
    };

    enum class MaterialModel : uint32_t
    {
        Legacy,
        Diffuse,
        Conductor,
        Dielectric,
        Count
    };

    // Isotropic GGX slope, not a perceptual roughness remapping. Zero is an
    // exact delta surface. Positive slopes below this supported numerical
    // range are rejected instead of silently changing their scattering measure.
    inline constexpr float MinimumMicrofacetSlope = 0.001f;
    inline constexpr float MinimumSubsurfaceMeanFreePath = 0.001f;
    inline constexpr float MaximumSubsurfaceMeanFreePath = 10000.0f;
    inline constexpr float MaximumSubsurfaceAnisotropy = 0.95f;

    // Shared data only; every backend still owns its scattering implementation.
    struct MaterialScattering
    {
        MaterialModel Model = MaterialModel::Legacy;
        float MicrofacetSlope = 0.2f;
        float ExteriorIndex = 1.0f;
        float InteriorIndex = 1.5f;
        // Analytic RGB constants relative to the exterior medium, not measured spectra.
        Float3 ConductorIndex = Float3(0.2f, 0.2f, 0.2f);
        Float3 ConductorExtinction = Float3(3.0f, 3.0f, 3.0f);
        // A homogeneous interior behind a dielectric boundary. Zero disables it.
        // Mean free path is world length; single-scattering albedo is the authored
        // material albedo times this scale, not a diffusion-profile reflectance.
        float SubsurfaceMeanFreePath = 0;
        float SubsurfaceAlbedoScale = 1;
        float SubsurfaceAnisotropy = 0;
        bool operator==(const MaterialScattering& other) const
        {
            return Model == other.Model && MicrofacetSlope == other.MicrofacetSlope
                && ExteriorIndex == other.ExteriorIndex && InteriorIndex == other.InteriorIndex
                && ConductorIndex.X == other.ConductorIndex.X && ConductorIndex.Y == other.ConductorIndex.Y
                && ConductorIndex.Z == other.ConductorIndex.Z
                && ConductorExtinction.X == other.ConductorExtinction.X
                && ConductorExtinction.Y == other.ConductorExtinction.Y
                && ConductorExtinction.Z == other.ConductorExtinction.Z
                && SubsurfaceMeanFreePath == other.SubsurfaceMeanFreePath
                && SubsurfaceAlbedoScale == other.SubsurfaceAlbedoScale
                && SubsurfaceAnisotropy == other.SubsurfaceAnisotropy;
        }
        bool operator!=(const MaterialScattering& other) const { return !(*this == other); }
    };
    ARVO_RENDERER_API bool ValidateMaterialScattering(const MaterialScattering& material, std::string& error);
    ARVO_RENDERER_API const char* MaterialModelKey(MaterialModel model);
    ARVO_RENDERER_API bool ParseMaterialModel(std::string_view key, MaterialModel& model);
    ARVO_RENDERER_API bool ValidateMaterialSupport(const MaterialScattering& material, BackendType backend,
        Algorithm algorithm, std::string& error);

    struct Material : MaterialScattering
    {
        Float3 Albedo = Float3(0.8f, 0.8f, 0.8f);
        Float3 AmbientColor = Float3(0.8f, 0.8f, 0.8f);
        Float3 SpecularColor = Float3(0.0f, 0.0f, 0.0f);
        Float3 Emission = Float3(0.0f, 0.0f, 0.0f);
        float Shininess = 1.0f;
        float Reflectivity = 0.0f;
        uint32_t DiffuseTextureIndex = InvalidRayTextureIndex;
        // The same emitter is sampled explicitly; integrators avoid counting
        // its mesh emission a second time after their direct-light estimator.
        bool SampledEmitter = false;
    };

    struct Triangle
    {
        Float3 Position0;
        Float3 Position1;
        Float3 Position2;
        Float3 Normal0;
        Float3 Normal1;
        Float3 Normal2;
        Float3 Color0 = Float3(1.0f, 1.0f, 1.0f);
        Float3 Color1 = Float3(1.0f, 1.0f, 1.0f);
        Float3 Color2 = Float3(1.0f, 1.0f, 1.0f);
        Float2 TexCoord0;
        Float2 TexCoord1;
        Float2 TexCoord2;
        uint32_t MaterialIndex = 0;
        bool SmoothNormals = false;
    };

    struct Sphere
    {
        Float3 Center;
        float Radius = 1.0f;
        uint32_t MaterialIndex = 0;
    };

    struct DirectionalLight
    {
        // A normalized vector from the shaded point towards the light.
        Float3 DirectionToLight = Normalize(Float3(-0.5f, 1.0f, 0.4f));
        Float3 Color = Float3(1.0f, 1.0f, 1.0f);
        float Intensity = 1.0f;
        bool CastShadows = true;
    };

    struct PointLight
    {
        Float3 Position;
        Float3 Color = Float3(1.0f, 1.0f, 1.0f);
        float Intensity = 1.0f;
        bool CastShadows = true;
    };

    struct RectAreaLight
    {
        Float3 Position;
        // Normalized direction in which the front face emits light.
        Float3 Direction = Float3(0.0f, -1.0f, 0.0f);
        Float3 Color = Float3(1.0f, 1.0f, 1.0f);
        float Intensity = 1.0f;
        // Full rectangle width and height in world units.
        Float2 Size = Float2(1.0f, 1.0f);
        bool CastShadows = true;
    };

    struct SceneGeometry
    {
        std::vector<Triangle> Triangles;
        std::vector<Sphere> Spheres;
    };

    struct GeometryInstance
    {
        uint32_t GeometryIndex = 0;
        // Row-major affine object-to-world matrix, applied to column vectors.
        std::array<float, 12> ObjectToWorld = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        MotionData::RigidMotion Motion;
    };

    struct Scene : SceneGeometry
    {
        std::vector<Material> Materials;
        // Materials reference this table by index so texels are shared by all
        // triangles that use the same image.
        std::vector<RayTexture> Textures;
        std::vector<std::shared_ptr<const SceneGeometry>> Geometries;
        std::vector<GeometryInstance> Instances;
        std::vector<DirectionalLight> DirectionalLights;
        std::vector<PointLight> PointLights;
        std::vector<RectAreaLight> RectAreaLights;
        // Linear environment colors. Equal defaults match the realtime clear.
        Float3 BackgroundTop = Float3(0.02f, 0.02f, 0.025f);
        Float3 BackgroundBottom = Float3(0.02f, 0.02f, 0.025f);
        float AmbientIntensity = 0.08f;
    };

    struct CameraLens
    {
        // World-space radius; zero preserves the pinhole camera exactly.
        float Radius = 0;
        // Axial distance to the plane of focus, not the optical focal length.
        float FocusDistance = 5;
        bool operator==(const CameraLens& other) const
        { return Radius == other.Radius && FocusDistance == other.FocusDistance; }
        bool operator!=(const CameraLens& other) const { return !(*this == other); }
    };
    inline constexpr float MinimumFocusDistance = 0.001f;
    ARVO_RENDERER_API bool ValidateCameraLens(const CameraLens& lens, std::string& error);

    struct Camera
    {
        Float3 Position = Float3(0.0f, 0.0f, 5.0f);
        Float3 Target = Float3(0.0f, 0.0f, 0.0f);
        Float3 Up = Float3(0.0f, 1.0f, 0.0f);
        float VerticalFieldOfViewDegrees = 60.0f;
        CameraLens Lens;
        MotionData::Exposure Motion;
    };

    ARVO_RENDERER_API bool ValidateCameraMotion(const MotionData::Exposure&, std::string& error);
    ARVO_RENDERER_API bool SupportsMotionBlur(Algorithm);
    ARVO_RENDERER_API bool ValidateRigidMotion(const MotionData::RigidMotion&, std::string& error);
    ARVO_RENDERER_API bool HasSceneMotion(const Scene&);
    ARVO_RENDERER_API bool ValidateSceneMotionSupport(Algorithm, const Scene&, std::string& error);
    ARVO_RENDERER_API bool ValidateCameraMotionSupport(Algorithm, const MotionData::Exposure&,
                                                      std::string& error);

    // A world-space homogeneous box; coefficients are inverse world units.
    // The boundary changes extinction only, without creating a material surface.
    struct MediumOptions
    {
        bool Enabled = false;
        Float3 Minimum = Float3(-1, -1, -1);
        Float3 Maximum = Float3(1, 1, 1);
        Float3 Absorption = Float3(0.05f, 0.05f, 0.05f);
        Float3 Scattering = Float3(0.15f, 0.15f, 0.15f);
        // Positive g favors continuation along the incident propagation direction.
        float Anisotropy = 0;
        bool operator==(const MediumOptions& other) const
        {
            const auto same = [](Float3 a, Float3 b) { return a.X == b.X && a.Y == b.Y && a.Z == b.Z; };
            return Enabled == other.Enabled && same(Minimum, other.Minimum) && same(Maximum, other.Maximum)
                && same(Absorption, other.Absorption) && same(Scattering, other.Scattering)
                && Anisotropy == other.Anisotropy;
        }
        bool operator!=(const MediumOptions& other) const { return !(*this == other); }
    };
    ARVO_RENDERER_API bool ValidateMediumOptions(Algorithm, const MediumOptions&, std::string& error);

    // Value one remains the native variance-only pilot phase. These independent
    // camera streams retain the same frame key instead of borrowing future keys.
    enum class CameraSampleStream : uint32_t { Production = 0, DenoisePilotA = 2, DenoisePilotB = 3 };
    enum class RadiometryMode : uint32_t { RGB, Spectral };
    ARVO_RENDERER_API bool ValidateRadiometry(Algorithm, RadiometryMode, std::string& error);
    ARVO_RENDERER_API const char* RadiometryKey(RadiometryMode);
    ARVO_RENDERER_API bool ParseRadiometry(const std::string&, RadiometryMode&);

    struct SampleBudgetImage
    {
        uint32_t Width = 0, Height = 0, FullWidth = 0, FullHeight = 0, TileX = 0, TileY = 0;
        uint32_t SamplingSeed = 0;
        Algorithm Integrator = Algorithm::PathTracing;
        RadiometryMode Radiometry = RadiometryMode::RGB;
        Camera View;
        std::vector<uint32_t> Counts;
    };

    struct RenderSettings
    {
        uint32_t Width = 640;
        uint32_t Height = 360;
        // Width/Height describe the tile; zero full dimensions select a full frame.
        uint32_t FullWidth = 0;
        uint32_t FullHeight = 0;
        uint32_t TileX = 0;
        uint32_t TileY = 0;
        Algorithm Integrator = Algorithm::RayTracing;
        RadiometryMode Radiometry = RadiometryMode::RGB;
        TransportOptions Transport;
        SampleSequence Sampling = SampleSequence::Independent;
        // Identifies a fresh observation of the same film. Keep this value fixed
        // across crops, dispatch batches and retries; zero preserves legacy streams.
        uint32_t SamplingSeed = 0;
        CameraSampleStream SampleStream = CameraSampleStream::Production;
        // Native production dispatch consumes this immutable, crop-matched map.
        std::shared_ptr<const SampleBudgetImage> ProductionBudgets;
        AdaptiveOptions Adaptive;
        MediumOptions Medium;
        MicropolygonOptions Micropolygons;
        std::shared_ptr<const MicropolygonPlan> DicingPlan;
        ReconstructionOptions Reconstruction;
        MetropolisOptions Metropolis;
        PhotonMappingOptions Photons;
        ProgressivePhotonOptions ProgressivePhotons;
        VertexConnectionMergingOptions Vcm;
        PointGiOptions PointGi;
        AccelerationType Acceleration = AccelerationType::Bvh;
        BvhBuildPolicy BvhPolicy = BvhBuildPolicy::Default;
        uint32_t SamplesPerPixel = 1;
        uint32_t MaxReflectionBounces = 1;
        // Zero selects the hardware concurrency reported by the standard library.
        uint32_t WorkerCount = 0;
        bool ShadowsEnabled = true;
        // Fraction of direct lighting removed when a shadow ray is occluded.
        float ShadowStrength = 1.0f;
        // Encodes linear radiance as exact display-sRGB when enabled.
        bool GammaCorrection = true;
    };

    inline uint32_t FullImageWidth(const RenderSettings& settings)
    {
        return settings.FullWidth == 0 ? settings.Width : settings.FullWidth;
    }

    inline uint32_t FullImageHeight(const RenderSettings& settings)
    {
        return settings.FullHeight == 0 ? settings.Height : settings.FullHeight;
    }

    ARVO_RENDERER_API bool ValidateSamplePlan(const RenderSettings&, std::string& error);
    inline bool ValidateTileSettings(const RenderSettings& settings, std::string& error)
    {
        const uint32_t width = FullImageWidth(settings);
        const uint32_t height = FullImageHeight(settings);
        if (width == 0 || height == 0 || width > 16384 || height > 16384
            || settings.TileX >= width || settings.TileY >= height
            || settings.Width == 0 || settings.Height == 0
            || settings.Width > width - settings.TileX
            || settings.Height > height - settings.TileY)
        {
            error = "Tile bounds must fit a nonempty image of at most 16384 x 16384.";
            return false;
        }
        if (!FindIntegrator(settings.Integrator))
        {
            error = "Unknown integration algorithm.";
            return false;
        }
        if (!ValidateTransportOptions(settings.Integrator, settings.Transport, error)) return false;
        if (!ValidateRadiometry(settings.Integrator, settings.Radiometry, error)) return false;
        if (!ValidateAdaptiveOptions(settings.Integrator, settings.Adaptive, settings.SamplesPerPixel, error)) return false;
        if (!ValidateSamplePlan(settings, error)) return false;
        if (!ValidateMediumOptions(settings.Integrator, settings.Medium, error)) return false;
        if (!ValidateMicropolygonOptions(settings.Micropolygons, error)) return false;
        if (!ValidateMetropolisOptions(settings.Metropolis, error)) return false;
        if (!ValidatePhotonMappingOptions(settings.Photons, error)) return false;
        if (!ValidateProgressivePhotonOptions(settings.ProgressivePhotons, error)) return false;
        if (!ValidateVertexConnectionMergingOptions(settings.Vcm, error))
            return false;
        if (!ValidatePointGiOptions(settings.PointGi, error)) return false;
        if (settings.Integrator == Algorithm::MetropolisLightTransport && settings.Sampling != SampleSequence::Independent)
        {
            error = "MLT uses independent primary samples and Markov mutations, not a camera sample sequence.";
            return false;
        }
        if (!ValidateAcceleration({settings.Acceleration, settings.BvhPolicy}, error)) return false;
        if (!FindSampler(settings.Sampling))
        {
            error = "Unknown sampling sequence.";
            return false;
        }
        return true;
    }

    class ARVO_RENDERER_API Image
    {
    public:
        Image() = default;
        Image(uint32_t width, uint32_t height);

        uint32_t GetWidth() const { return m_Width; }
        uint32_t GetHeight() const { return m_Height; }
        bool IsEmpty() const { return m_Pixels.empty(); }
        // Top-left RGBA8. RGB is display-sRGB when GammaCorrection is enabled.
        const std::vector<uint8_t>& GetPixelsRGBA8() const { return m_Pixels; }
        std::vector<uint8_t>& GetPixelsRGBA8() { return m_Pixels; }
        const uint8_t* GetPixel(uint32_t x, uint32_t y) const;
        bool SavePPM(const std::string& path, std::string* error = nullptr) const;

    private:
        uint32_t m_Width = 0;
        uint32_t m_Height = 0;
        std::vector<uint8_t> m_Pixels;
    };

    enum class BackendType : uint8_t
    {
        CPU,
        // DXR supports both offline rendering and Renderer realtime preview.
        DXR,
        CUDA,
        ComputeShader,
        Metal
    };

    struct BackendInfo
    {
        BackendType Type = BackendType::CPU;
        std::string Name;
        std::string Description;
        bool IsNative = false;
        bool IsHardwareAccelerated = false;
        bool IsAvailable = true;
    };

    enum class RenderStatus : uint8_t
    {
        Completed,
        Cancelled,
        Failed
    };

    struct RenderStatistics
    {
        uint64_t PrimaryRays = 0;
        uint64_t PilotRays = 0;
        uint64_t ShadowRays = 0;
        uint64_t ReflectionRays = 0;
        uint64_t PrimitiveTests = 0;
        uint64_t CompletedPixels = 0;
        double BuildMilliseconds = 0.0;
        double RenderMilliseconds = 0.0;
    };

    struct FilmPixel
    {
        Float3 MeanRadiance;
        // Sum of squared luminance deviations, before any display transform.
        // M2/(n-1) is sample variance for n>1. Dividing again by n estimates
        // variance of the mean for independent samples; QMC needs care. MLT
        // records complete mutation passes, which remain correlated. Their M2
        // is descriptive dispersion, not an independent-sample error estimate.
        // PPM writes a final density estimate, not a mean of pass estimates;
        // its M2 is unavailable (zero), and SampleCount records completed passes.
        // VCM averages changing-bandwidth joint estimates with shared light paths;
        // its M2 is likewise unavailable, and SampleCount records completed passes.
        float LuminanceM2 = 0.0f;
        uint32_t SampleCount = 0;
    };

    struct LinearFilm
    {
        uint32_t Width = 0;
        uint32_t Height = 0;
          // Top-left row order, matching Output. No tone mapping, gamma or clamp.
          // Spectral observations may contain signed linear RGB basis coordinates.
        std::vector<FilmPixel> Pixels;

        LinearFilm() = default;
        LinearFilm(uint32_t width, uint32_t height)
            : Width(width), Height(height), Pixels(static_cast<size_t>(width) * height) {}
    };

    // Filtered RGB is a biased image estimate, with no invented moments/counts.
    struct ReconstructedImage
    {
        uint32_t Width = 0, Height = 0;
        std::vector<Float3> Pixels;
        Image Display;
    };

    struct RenderResult
    {
        RenderStatus Status = RenderStatus::Failed;
        Image Output;
        LinearFilm Film;
        // Same extent as Film, empty when reconstruction is disabled.
        std::vector<Float3> ReconstructedRadiance;
        RenderStatistics Statistics;
        std::string Error;
    };

    inline constexpr uint32_t InvalidRayPrimitiveIndex = std::numeric_limits<uint32_t>::max();

    // Direct native traversal diagnostics. Primitive indices address triangles,
    // followed by spheres, then each instance's triangles followed by spheres
    // in scene instance order. Shared objects get a distinct ID range per instance.
    // t always belongs to the original, unnormalized ray.
    // The query interval is (MinDistance, MaxDistance). No shading is performed.
    // Open bounds match DXR's fixed-function triangle intersection convention:
    // https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#ray-extents
    // Exact coincident hits may select either primitive; ordering is backend-owned.
    struct RayQueryHit
    {
        static constexpr uint32_t HitFlag = 1;
        static constexpr uint32_t TexCoordFlag = 2;
        static constexpr uint32_t FrontFaceFlag = 4;
        // Unit shading normal facing the incident ray; never use it to select a medium.
        Float3 Normal;
        float Distance = std::numeric_limits<float>::max();
        Float3 Color = Float3(1.0f, 1.0f, 1.0f);
        uint32_t PrimitiveIndex = InvalidRayPrimitiveIndex;
        Float2 TexCoord;
        uint32_t MaterialIndex = 0;
        uint32_t Flags = 0;
        // Oriented unit geometric normal: triangle winding or sphere exterior,
        // transported by the inverse transpose without a determinant sign flip.
        Float3 GeometricNormal;
        uint32_t Reserved = 0;
        bool HasHit() const { return (Flags & HitFlag) != 0; }
        bool HasTexCoord() const { return (Flags & TexCoordFlag) != 0; }
        // Strictly entering: dot(GeometricNormal, ray.Direction) < 0; tangency is false.
        bool IsFrontFace() const { return (Flags & FrontFaceFlag) != 0; }
    };
    static_assert(sizeof(RayQueryHit) == 64 && offsetof(RayQueryHit, Distance) == 12
        && offsetof(RayQueryHit, PrimitiveIndex) == 28 && offsetof(RayQueryHit, TexCoord) == 32
        && offsetof(RayQueryHit, MaterialIndex) == 40 && offsetof(RayQueryHit, Flags) == 44
        && offsetof(RayQueryHit, GeometricNormal) == 48 && offsetof(RayQueryHit, Reserved) == 60,
        "Native ray-query hit ABI mismatch");

    struct RayQueryResult
    {
        RenderStatus Status = RenderStatus::Failed;
        std::vector<RayQueryHit> Hits;
        std::string Error;
    };
    // Deterministic pinhole geometry probes, independent of noisy transport.
    // These are not sample-averaged AOVs or a representation of optical paths.
    struct GeometryGuide
    {
        static constexpr uint32_t Surface = 1, Background = 2, FrontFace = 4;
        Float3 Position;
        float CameraDepth = 0;
        Float3 GeometricNormal;
        float RayDistance = 0;
        Float3 Direction;
        uint32_t Flags = 0;
        uint32_t PrimitiveIndex = InvalidRayPrimitiveIndex;
        uint32_t MaterialIndex = 0;
        uint32_t Reserved[2]{};
    };
    static_assert(sizeof(GeometryGuide) == 64 && offsetof(GeometryGuide, Direction) == 32 &&
                      offsetof(GeometryGuide, PrimitiveIndex) == 48,
                  "Geometry guide ABI mismatch");
    struct GeometryGuideImage
    {
        uint32_t Width = 0, Height = 0, FullWidth = 0, FullHeight = 0, TileX = 0, TileY = 0;
        // Finite apertures integrate several surfaces; a central probe alone
        // must never authorize temporal history across that optical footprint.
        bool PinholeCorrespondence = false;
        std::vector<GeometryGuide> Pixels;
        // Local spatial reconstruction only. A negative triplet marks an ineligible footprint.
        std::vector<Float3> DiffuseAlbedo;
    };
    struct GeometryGuideResult
    {
        RenderStatus Status = RenderStatus::Failed;
        GeometryGuideImage Image;
        std::string Error;
    };
    inline constexpr uint32_t MaximumRayQueryBatch = 1u << 20;
    ARVO_RENDERER_API bool ValidateRayQueries(const std::vector<Ray>& rays, std::string& error);

    class ARVO_RENDERER_API CancellationToken
    {
    public:
        CancellationToken() = default;
        // Interoperability with a caller-owned asynchronous cancellation source.
        static CancellationToken FromSharedState(std::shared_ptr<std::atomic<bool>> state)
        { return CancellationToken(std::move(state)); }
        bool IsCancellationRequested() const;

    private:
        explicit CancellationToken(std::shared_ptr<std::atomic<bool>> state) : m_State(std::move(state)) {}
        std::shared_ptr<std::atomic<bool>> m_State;
        friend class CancellationSource;
    };

    class ARVO_RENDERER_API CancellationSource
    {
    public:
        CancellationSource();
        CancellationToken GetToken() const;
        void Cancel() const;
        bool IsCancellationRequested() const;

    private:
        std::shared_ptr<std::atomic<bool>> m_State;
    };

    class ARVO_RENDERER_API RenderJob
    {
    public:
        // Backend queues use this constructor to bind a future to its cancellation state.
        RenderJob(CancellationSource cancellation, std::future<RenderResult> future);
        RenderJob(RenderJob&&) noexcept;
        RenderJob& operator=(RenderJob&&) noexcept;
        ~RenderJob();

        RenderJob(const RenderJob&) = delete;
        RenderJob& operator=(const RenderJob&) = delete;

        void Cancel() const;
        bool IsReady() const;
        void Wait() const;
        RenderResult GetResult();

    private:
        CancellationSource m_Cancellation;
        std::future<RenderResult> m_Future;
    };

    struct TemporalFrame;
    struct TemporalHistory;
    struct TemporalOptions;
    struct TemporalResult;
    struct MicropolygonResult;
    class ARVO_RENDERER_API IRayTracingRHI
    {
    public:
        virtual ~IRayTracingRHI() = default;
        virtual MicropolygonResult PrepareMicropolygons(
            const Scene&, const Camera&, const RenderSettings&,
            const CancellationToken& = CancellationToken()) const;
        virtual TemporalResult ReconstructTemporal(const TemporalFrame&, const TemporalHistory*,
                                                   const TemporalOptions&,
                                                   const CancellationToken& = CancellationToken()) const;
        virtual BackendInfo GetBackendInfo() const = 0;
        virtual GeometryGuideResult CaptureGeometryGuides(
            const Scene&, const Camera&, const RenderSettings&,
            const CancellationToken& = CancellationToken()) const
        {
            return {RenderStatus::Failed, {}, "Geometry guides are not implemented by this backend"};
        }
        virtual RayQueryResult QueryRays(const Scene& scene, const std::vector<Ray>& rays,
            AccelerationOptions acceleration = AccelerationType::Bvh,
            const CancellationToken& cancellation = CancellationToken()) const = 0;
        virtual RenderResult Render(
            const Scene& scene,
            const Camera& camera,
            const RenderSettings& settings,
            const CancellationToken& cancellation = CancellationToken()) const = 0;
        virtual RenderJob RenderAsync(Scene scene, Camera camera, RenderSettings settings) const = 0;
        // Shared scenes are immutable snapshots used as upload-cache identities.
        // Publish a new owner after editing geometry, materials or lighting.
        virtual RenderJob RenderAsync(
            std::shared_ptr<const Scene> scene,
            Camera camera,
            RenderSettings settings) const = 0;
        virtual void WaitIdle() const = 0;
    };

}
}
