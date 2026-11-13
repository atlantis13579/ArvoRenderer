#include "Renderer/SubsurfaceValidation.h"
#include "MetalRayTracingRHI.h"
#include "Renderer/Reconstruction.h"
#include "Renderer/AdaptiveReconstruction.h"
#include "Renderer/ParticipatingMedia.h"
#include "Renderer/SpectralSceneData.h"
#include "Renderer/GeometryGuides.h"
#include "Renderer/TemporalReconstruction.h"
#include "Renderer/MicropolygonPreparation.h"
#include <type_traits>

#if defined(__APPLE__)

#include "Backends/Metal/Limits.h"
#include "Backends/Metal/ScenePacking.h"
#include "Backends/Metal/BidirectionalLayout.h"
#include "Backends/Metal/MetropolisLayout.h"
#include "Backends/Metal/PhotonLayout.h"
#include "Backends/Metal/ProgressivePhotonLayout.h"
#include "Backends/Metal/PointGiGeometry.h"
#include "Backends/Metal/VertexConnectionMergingFrame.h"
#include "Maths/ColorSpace.h"
#include "Renderer/EmitterData.h"
#include "MetalShaderSources.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace Arvo
{
namespace RayTracing
{
namespace
{

using namespace MetalScene;
static_assert(MetalBdpt::MaximumReflections == MetalLimits::MaximumReflectionBounces,
              "Metal BDPT path storage must cover the validated depth range.");
constexpr NSUInteger InstanceRecordBinding = 13u, InstanceNodeBinding = 14u, InstanceErrorBinding = 15u;

constexpr std::uint32_t ThreadsPerGroup = 256u;
constexpr std::uint32_t PixelChunkSize = 16384u;
constexpr float Pi = 3.14159265358979323846f;

struct QueryRay
{
    Vec4 OriginMinimum;
    Vec4 DirectionMaximum;
};
static_assert(sizeof(QueryRay) == 32u && sizeof(RayQueryHit) == 64u
    && offsetof(RayQueryHit, Flags) == 44u && offsetof(RayQueryHit, GeometricNormal) == 48u,
    "Metal query buffer ABI mismatch.");

struct alignas(16) FrameConstants
{
    Vec3 CameraPosition;
    std::uint32_t Width = 0;
    Vec3 CameraForward;
    std::uint32_t Height = 0;
    Vec3 CameraRight;
    std::uint32_t TotalSamples = 0;
    Vec3 CameraUp;
    std::uint32_t MaxReflectionBounces = 0;
    float HalfWidth = 0.0f;
    float HalfHeight = 0.0f;
    std::uint32_t MaterialCount = 0;
    std::uint32_t LightCount = 0;
    Vec3 BackgroundTop;
    float AmbientIntensity = 0.0f;
    Vec3 BackgroundBottom;
    std::uint32_t NodeCount = 0;
    std::uint32_t PrimitiveRefCount = 0;
    std::uint32_t TriangleCount = 0;
    std::uint32_t SphereCount = 0;
    float ShadowStrength = 0.0f;
    std::uint32_t PixelBegin = 0;
    std::uint32_t PixelCount = 0;
    std::uint32_t SampleBegin = 0;
    std::uint32_t SampleCount = 0;
    std::uint32_t GammaCorrection = 0;
    std::uint32_t GroupCount = 0;
    std::uint32_t TextureCount = 0;
    std::uint32_t TextureTexelCount = 0;
    std::uint32_t FullWidth = 0;
    std::uint32_t FullHeight = 0;
    std::uint32_t TileX = 0;
    std::uint32_t TileY = 0;
    std::uint32_t Integrator = 0;
    std::uint32_t TransportFlags = 0;
    std::uint32_t RouletteStartDepth = 0;
    std::uint32_t SamplingSequence = 0;
    std::uint32_t AdaptiveEnabled = 0;
    std::uint32_t AdaptivePilotSamples = 0;
    std::uint32_t AdaptiveMinimumSamples = 0;
    std::uint32_t AdaptivePhase = 0;
    float AdaptiveRelativeError = 0.0f;
    float AdaptiveAbsoluteError = 0.0f;
    std::uint32_t Acceleration = 0;
    std::uint32_t InstanceNodeCount = 0;
    float LensRadius = 0;
    float FocusDistance = 0;
    std::uint32_t SamplingSeed = 0, LensPadding1 = 0;
};

struct Counter64Parts
{
    std::uint32_t Low = 0;
    std::uint32_t High = 0;

    [[nodiscard]] std::uint64_t Value() const noexcept
    {
        return static_cast<std::uint64_t>(Low)
            | (static_cast<std::uint64_t>(High) << 32u);
    }
};

struct GroupStatistics
{
    Counter64Parts PrimaryRays;
    Counter64Parts ShadowRays;
    Counter64Parts ReflectionRays;
    Counter64Parts PrimitiveTests;
};

static_assert(sizeof(Vec3) == 12u, "Metal packed float3 host ABI mismatch.");
static_assert(sizeof(Vec4) == 16u, "Metal float4 host ABI mismatch.");
static_assert(sizeof(PackedAccelerationNode) == 48u, "Metal acceleration node ABI mismatch.");
static_assert(offsetof(PackedAccelerationNode, SplitPosition) == 32u
    && offsetof(PackedAccelerationNode, AboveChild) == 40u, "Metal kd-tree field ABI mismatch.");
static_assert(offsetof(PackedAccelerationNode, SecondPlane) == 44u, "Metal BIH field ABI mismatch.");
static_assert(offsetof(PackedAccelerationNode, FirstOrEscape) == 12u, "Metal BVH field ABI mismatch.");
static_assert(offsetof(PackedAccelerationNode, PrimitiveCount) == 28u, "Metal BVH field ABI mismatch.");
static_assert(sizeof(PackedPrimitiveRef) == 4u, "Metal primitive reference ABI mismatch.");
static_assert(sizeof(PackedTriangle) == 192u, "Metal triangle ABI mismatch.");
static_assert(offsetof(PackedTriangle, TexCoord01) == 144u, "Metal triangle field ABI mismatch.");
static_assert(offsetof(PackedTriangle, TexCoord2) == 160u, "Metal triangle field ABI mismatch.");
static_assert(offsetof(PackedTriangle, MaterialIndex) == 176u, "Metal triangle field ABI mismatch.");
static_assert(sizeof(PackedSphere) == 32u, "Metal sphere ABI mismatch.");
static_assert(sizeof(PackedMaterial) == 128u, "Metal material ABI mismatch.");
static_assert(
    offsetof(PackedMaterial, AmbientShininess) == 16u
        && offsetof(PackedMaterial, Specular) == 32u
        && offsetof(PackedMaterial, Emission) == 48u
        && offsetof(PackedMaterial, DiffuseTextureIndex) == 64u,
    "Metal material field ABI mismatch.");
static_assert(
    sizeof(Detail::PackedRayTextureDescriptor) == 16u,
    "Metal texture descriptor ABI mismatch.");
static_assert(sizeof(PackedLight) == 64u, "Metal light ABI mismatch.");
static_assert(offsetof(PackedLight, Position) == 0u, "Metal light position ABI mismatch.");
static_assert(offsetof(PackedLight, Type) == 12u, "Metal light type ABI mismatch.");
static_assert(offsetof(PackedLight, Direction) == 16u, "Metal light direction ABI mismatch.");
static_assert(offsetof(PackedLight, Intensity) == 28u, "Metal light intensity ABI mismatch.");
static_assert(offsetof(PackedLight, Color) == 32u, "Metal light color ABI mismatch.");
static_assert(offsetof(PackedLight, CastShadows) == 44u, "Metal light shadow ABI mismatch.");
static_assert(offsetof(PackedLight, Width) == 48u, "Metal light width ABI mismatch.");
static_assert(offsetof(PackedLight, Height) == 52u, "Metal light height ABI mismatch.");
static_assert(offsetof(PackedLight, Padding0) == 56u, "Metal light padding ABI mismatch.");
static_assert(offsetof(PackedLight, Padding1) == 60u, "Metal light padding ABI mismatch.");
static_assert(sizeof(FrameConstants) == 240u, "Metal frame constant ABI mismatch.");
static_assert(offsetof(FrameConstants, Width) == 12u, "Metal camera constant ABI mismatch.");
static_assert(offsetof(FrameConstants, BackgroundBottom) == 96u, "Metal environment constant ABI mismatch.");
static_assert(offsetof(FrameConstants, ShadowStrength) == 124u, "Metal shadow constant ABI mismatch.");
static_assert(offsetof(FrameConstants, PixelBegin) == 128u, "Metal dispatch constant ABI mismatch.");
static_assert(offsetof(FrameConstants, TextureCount) == 152u, "Metal texture constant ABI mismatch.");
static_assert(offsetof(FrameConstants, FullWidth) == 160u, "Metal tile constant ABI mismatch.");
static_assert(offsetof(FrameConstants, Integrator) == 176u, "Metal integrator constant ABI mismatch.");
static_assert(offsetof(FrameConstants, AdaptiveEnabled) == 192u, "Metal adaptive constant ABI mismatch.");
static_assert(offsetof(FrameConstants, Acceleration) == 216u, "Metal acceleration constant ABI mismatch.");
static_assert(offsetof(FrameConstants, InstanceNodeCount) == 220u, "Metal instance constant ABI mismatch.");
static_assert(offsetof(FrameConstants, LensRadius) == 224u && offsetof(FrameConstants, FocusDistance) == 228u,
    "Metal lens constant ABI mismatch.");
static_assert(sizeof(GroupStatistics) == 32u, "Metal statistics ABI mismatch.");

[[nodiscard]] Vec3 ToVec3(const Float3& value)
{
    return {value.X, value.Y, value.Z};
}

[[nodiscard]] Vec3 Add(const Vec3& left, const Vec3& right)
{
    return {left.X + right.X, left.Y + right.Y, left.Z + right.Z};
}

[[nodiscard]] Vec3 Subtract(const Vec3& left, const Vec3& right)
{
    return {left.X - right.X, left.Y - right.Y, left.Z - right.Z};
}

[[nodiscard]] Vec3 Multiply(const Vec3& value, float scalar)
{
    return {value.X * scalar, value.Y * scalar, value.Z * scalar};
}

[[nodiscard]] float DotVector(const Vec3& left, const Vec3& right)
{
    return left.X * right.X + left.Y * right.Y + left.Z * right.Z;
}

[[nodiscard]] Vec3 CrossVector(const Vec3& left, const Vec3& right)
{
    return {
        left.Y * right.Z - left.Z * right.Y,
        left.Z * right.X - left.X * right.Z,
        left.X * right.Y - left.Y * right.X};
}

[[nodiscard]] Vec3 NormalizeVector(const Vec3& value)
{
    const float lengthSquared = DotVector(value, value);
    if (lengthSquared <= 1.0e-20f)
    {
        return {};
    }
    return Multiply(value, 1.0f / std::sqrt(lengthSquared));
}

[[nodiscard]] bool IsFinite(const Float3& value)
{
    return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
}

[[nodiscard]] bool IsNonNegative(const Float3& value)
{
    return value.X >= 0.0f && value.Y >= 0.0f && value.Z >= 0.0f;
}

[[nodiscard]] bool IsFinite(const Float2& value)
{
    return std::isfinite(value.X) && std::isfinite(value.Y);
}

[[nodiscard]] bool IsFinite(const Vec3& value)
{
    return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
}

[[nodiscard]] bool ValidateInput(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings,
    std::string& error, bool validateMotionSchedule = true)
{
    if (!MetalInstances::Validate(scene, error)) return false;
    if (!ValidateCameraLens(camera.Lens, error) ||
                (validateMotionSchedule && !ValidateSubsurfaceScene(scene, settings, error)) ||
        !ValidateCameraMotionSupport(settings.Integrator, camera.Motion, error) ||
                (validateMotionSchedule && !ValidateSceneMotionSupport(settings.Integrator, scene, error)) ||
        !ValidateCameraSamplePlan(camera, settings, error))
        return false;
    if (!ValidateTileSettings(settings, error))
        return false;
    if (!ValidateIntegrator(BackendType::Metal, settings.Integrator, error)) return false;
    if (settings.Width == 0u || settings.Height == 0u
        || settings.Width > 16384u || settings.Height > 16384u)
    {
        error = "Metal image dimensions must be in the range [1, 16384].";
        return false;
    }
    if (settings.SamplesPerPixel == 0u || settings.SamplesPerPixel > 4096u
        || settings.MaxReflectionBounces > MetalLimits::MaximumReflectionBounces)
    {
        error = "Metal sample or reflection settings exceed their safety limits.";
        return false;
    }
    if (!std::isfinite(settings.ShadowStrength)
        || settings.ShadowStrength < 0.0f || settings.ShadowStrength > 1.0f)
    {
        error = "Metal shadow strength must be finite and in the range [0, 1].";
        return false;
    }
    if (!IsFinite(camera.Position) || !IsFinite(camera.Target) || !IsFinite(camera.Up)
        || !std::isfinite(camera.VerticalFieldOfViewDegrees)
        || camera.VerticalFieldOfViewDegrees <= 1.0f
        || camera.VerticalFieldOfViewDegrees >= 179.0f)
    {
        error = "Metal camera values are invalid.";
        return false;
    }
    const Vec3 cameraDirection = Subtract(ToVec3(camera.Target), ToVec3(camera.Position));
    const Vec3 cameraUp = ToVec3(camera.Up);
    const Vec3 frameCross = CrossVector(
        NormalizeVector(cameraDirection),
        NormalizeVector(cameraUp));
    const float directionLengthSquared = DotVector(cameraDirection, cameraDirection);
    const float upLengthSquared = DotVector(cameraUp, cameraUp);
    const float frameAreaSquared = DotVector(frameCross, frameCross);
    if (!std::isfinite(directionLengthSquared) || !std::isfinite(upLengthSquared)
        || !std::isfinite(frameAreaSquared)
        || directionLengthSquared < 1.0e-10f
        || upLengthSquared < 1.0e-10f
        || frameAreaSquared < 1.0e-10f)
    {
        error = "Metal camera vectors do not form a valid frame.";
        return false;
    }
    if ((!scene.Triangles.empty() || !scene.Spheres.empty()) && scene.Materials.empty())
    {
        error = "Metal geometry requires at least one material.";
        return false;
    }
    if (scene.Triangles.size() >= SphereBit || scene.Spheres.size() >= SphereBit
        || scene.Triangles.size() + scene.Spheres.size() >= SphereBit
        || scene.Materials.size() > std::numeric_limits<std::uint32_t>::max()
        || scene.Textures.size() > std::numeric_limits<std::uint32_t>::max())
    {
        error = "Metal scene counts exceed the 31-bit primitive layout.";
        return false;
    }
    std::uint64_t textureTexelCount = 0u;
    for (const RayTexture& texture : scene.Textures)
    {
        if (!texture.IsValid())
        {
            continue;
        }
        textureTexelCount +=
            static_cast<std::uint64_t>(texture.Width) * texture.Height;
        if (textureTexelCount > std::numeric_limits<std::uint32_t>::max())
        {
            error = "Metal texture atlas exceeds its 32-bit texel layout.";
            return false;
        }
    }
    const std::size_t maximumLightCount =
        std::numeric_limits<std::uint32_t>::max();
    if (scene.DirectionalLights.size() > maximumLightCount
        || scene.PointLights.size() > maximumLightCount
        || scene.RectAreaLights.size() > maximumLightCount
        || scene.PointLights.size()
            > maximumLightCount - scene.DirectionalLights.size()
        || scene.RectAreaLights.size()
            > maximumLightCount
                - scene.DirectionalLights.size()
                - scene.PointLights.size())
    {
        error = "Metal light counts exceed the 32-bit light layout.";
        return false;
    }
    if (!IsFinite(scene.BackgroundTop) || !IsFinite(scene.BackgroundBottom)
        || !std::isfinite(scene.AmbientIntensity) || scene.AmbientIntensity < 0.0f)
    {
        error = "Metal background and ambient values are invalid.";
        return false;
    }
    for (const Material& material : scene.Materials)
    {
        if (!IsFinite(material.Albedo) || !IsFinite(material.AmbientColor)
            || !IsFinite(material.SpecularColor) || !IsFinite(material.Emission)
            || !std::isfinite(material.Shininess)
            || !std::isfinite(material.Reflectivity)
            || !IsNonNegative(material.AmbientColor)
            || !IsNonNegative(material.SpecularColor)
            || material.Shininess < 0.0f)
        {
            error = "Metal material values must be finite; ambient, specular, and shininess must be non-negative.";
            return false;
        }
    }
    if (!ValidateGeometry(scene, scene.Materials.size(), error)) return false;
    for (const auto& geometry : scene.Geometries)
        if (!ValidateGeometry(*geometry, scene.Materials.size(), error)) return false;
    for (const DirectionalLight& light : scene.DirectionalLights)
    {
        const Vec3 direction = ToVec3(light.DirectionToLight);
        const float directionLengthSquared = DotVector(direction, direction);
        if (!IsFinite(light.DirectionToLight) || !IsFinite(light.Color)
            || !std::isfinite(light.Intensity) || light.Intensity < 0.0f
            || !std::isfinite(directionLengthSquared)
            || directionLengthSquared < 1.0e-12f)
        {
            error = "Metal directional light data is invalid.";
            return false;
        }
    }
    for (const PointLight& light : scene.PointLights)
    {
        if (!IsFinite(light.Position) || !IsFinite(light.Color)
            || !std::isfinite(light.Intensity) || light.Intensity < 0.0f)
        {
            error =
                "Metal point light values must be finite with non-negative intensity.";
            return false;
        }
    }
    for (const RectAreaLight& light : scene.RectAreaLights)
    {
        const Vec3 direction = ToVec3(light.Direction);
        const float directionLengthSquared = DotVector(direction, direction);
        if (!IsFinite(light.Position) || !IsFinite(light.Direction)
            || !IsFinite(light.Color) || !IsFinite(light.Size)
            || !std::isfinite(light.Intensity) || light.Intensity < 0.0f
            || !std::isfinite(directionLengthSquared)
            || directionLengthSquared < 1.0e-12f
            || light.Size.X <= 0.0f || light.Size.Y <= 0.0f)
        {
            error =
                "Metal rect area light values must be finite with a non-zero "
                "direction, positive size, and non-negative intensity.";
            return false;
        }
        const Vec3 normalizedDirection = NormalizeVector(direction);
        const Vec3 helper = std::fabs(normalizedDirection.Y) < 0.999f
            ? Vec3{0.0f, 1.0f, 0.0f}
            : Vec3{1.0f, 0.0f, 0.0f};
        const Vec3 tangent =
            NormalizeVector(CrossVector(helper, normalizedDirection));
        const Vec3 bitangent =
            NormalizeVector(CrossVector(normalizedDirection, tangent));
        const Vec3 halfTangent = Multiply(tangent, light.Size.X * 0.5f);
        const Vec3 halfBitangent = Multiply(bitangent, light.Size.Y * 0.5f);
        const Vec3 position = ToVec3(light.Position);
        if (!IsFinite(Add(Add(position, halfTangent), halfBitangent))
            || !IsFinite(Subtract(Add(position, halfTangent), halfBitangent))
            || !IsFinite(Add(Subtract(position, halfTangent), halfBitangent))
            || !IsFinite(Subtract(Subtract(position, halfTangent), halfBitangent)))
        {
            error = "Metal rect area light bounds must remain finite.";
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::string ErrorDescription(NSError* error, const char* fallback)
{
    if (error != nil)
    {
        const char* description = [[error localizedDescription] UTF8String];
        if (description != nullptr)
        {
            return description;
        }
    }
    return fallback;
}

void ReleaseObject(id object)
{
    if (object != nil)
    {
        [object release];
    }
}

struct CommandCompletionState
{
    std::mutex Mutex;
    std::condition_variable Condition;
    bool Completed = false;
};

} // namespace

struct MetalRayTracingRHI::Impl
{
    struct AvailabilitySnapshot
    {
        bool Available = false;
        bool Terminal = false;
        std::string Error;
        std::string DeviceName;
    };

    struct Buffer
    {
        Buffer() = default;

        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;

        Buffer(Buffer&& other) noexcept
            : Handle(other.Handle)
            , LogicalSize(other.LogicalSize)
        {
            other.Handle = nil;
            other.LogicalSize = 0u;
        }

        Buffer& operator=(Buffer&& other) noexcept
        {
            if (this != &other)
            {
                Reset();
                Handle = other.Handle;
                LogicalSize = other.LogicalSize;
                other.Handle = nil;
                other.LogicalSize = 0u;
            }
            return *this;
        }

        ~Buffer()
        {
            Reset();
        }

        void Reset()
        {
            ReleaseObject(Handle);
            Handle = nil;
            LogicalSize = 0u;
        }

        id<MTLBuffer> Handle = nil;
        std::size_t LogicalSize = 0u;
    };

    Impl()
    {
        Initialize();
    }

    ~Impl()
    {
        MetalQueue.WaitIdle();
        Shutdown();
    }

    [[nodiscard]] AvailabilitySnapshot GetAvailability() const
    {
        std::lock_guard<std::mutex> lock(AvailabilityMutex);
        return {Available, Terminal, AvailabilityError, DeviceName};
    }

    void SetUnavailable(std::string error)
    {
        std::lock_guard<std::mutex> lock(AvailabilityMutex);
        Available = false;
        if (!Terminal)
        {
            AvailabilityError = std::move(error);
        }
    }

    void SetAvailable(std::string deviceName)
    {
        std::lock_guard<std::mutex> lock(AvailabilityMutex);
        Available = true;
        Terminal = false;
        AvailabilityError.clear();
        DeviceName = std::move(deviceName);
    }

    void MarkTerminal(std::string error) const
    {
        std::lock_guard<std::mutex> lock(AvailabilityMutex);
        Available = false;
        Terminal = true;
        AvailabilityError = std::move(error);
    }

    RenderJob Submit(
        const Scene* scene,
        std::shared_ptr<const Scene> owner,
        Camera camera,
        RenderSettings settings,
        CancellationToken external = CancellationToken()) const
    {
        return MetalQueue.Submit(
            [this,
             scene,
             owner = std::move(owner),
             camera,
             settings,
             external](const CancellationToken& queued) mutable
            {
                if (scene == nullptr)
                {
                    return Detail::MakeFailedResult(
                        "The Metal ray tracing submission has no scene.");
                }
                const Detail::CancellationProbe cancellation =
                    Detail::CombineCancellation(queued, external);
                if (cancellation())
                {
                    return Detail::MakeCancelledResult();
                }
                const AvailabilitySnapshot availability = GetAvailability();
                if (!availability.Available)
                {
                    return Detail::MakeFailedResult(
                        std::string("Metal compute ray tracing is unavailable: ")
                        + availability.Error);
                }
                @autoreleasepool
                {
                    if (settings.Micropolygons.Enabled)
                    {
                        auto diced = Detail::PrepareMicropolygons(
                            *scene, camera, settings, cancellation,
                            [&](const auto& controls, const auto& patches, const auto& ranges, auto& words,
                                auto& error) {
                                return MicropolygonPassNative(controls, patches, ranges, words, cancellation,
                                                              error);
                            });
                        if (diced.Status != RenderStatus::Completed)
                            return diced.Status == RenderStatus::Cancelled
                                       ? Detail::MakeCancelledResult()
                                       : Detail::MakeFailedResult(diced.Error);
                        owner = std::move(diced.Geometry);
                        scene = owner.get();
                        settings.Micropolygons.Enabled = false;
                        settings.DicingPlan.reset();
                    }
                    Detail::SpatialGuideCache spatialGuides;
                    return Detail::RenderWithDenoisedSampling(
                        *scene, camera, settings, cancellation, [&](const RenderSettings &expanded)
                        { return RenderNative(*scene, owner, camera, expanded, cancellation); },
                        [&](const RenderSettings &raster, const LinearFilm &film, ReconstructedImage &image,
                            std::string &error)
                        {
                            const GeometryGuideImage *guides = nullptr;
                            if (settings.Reconstruction.GeometryGuidance)
                            {
                                guides = spatialGuides.Get(
                                    raster,
                                    [&] { return CaptureGuidesNative(*scene, camera, raster, cancellation); },
                                    error);
                                if (!guides)
                                    return false;
                            }
                            return ReconstructNative(film, settings, guides, cancellation, image, error);
                        },
                        [&](const auto &input, auto controls, auto &counts, auto &error)
                        { return AllocateDenoisedNative(input, controls, cancellation, counts, error); });
                }
            });
    }

    bool Initialize()
    {
        @autoreleasepool
        {
            Device = MTLCreateSystemDefaultDevice();
            if (Device == nil)
            {
                SetUnavailable("No Metal device is available.");
                return false;
            }
            CommandQueue = [Device newCommandQueue];
            if (CommandQueue == nil)
            {
                SetUnavailable(
                    "The Metal device could not create a command queue.");
                return false;
            }

            if (!CreatePipeline(Detail::MetalWhittedSource, @"ArvoTracePixels", WhittedPipeline)
                || !CreatePipeline(Detail::MetalWhittedSource, @"ArvoQueryRays", QueryPipeline)
                || !CreatePipeline(Detail::MetalPathTracingSource, @"ArvoTracePixels", PathTracingPipeline)
                || !CreatePipeline(Detail::MetalDistributedRaySource, @"ArvoTracePixels", DistributedRayPipeline)
                || !CreatePipeline(Detail::MetalFilmSource, @"ArvoResolvePixels", ResolvePipeline))
                return false;

            const char* name = [[Device name] UTF8String];
            SetAvailable(name != nullptr ? name : "Unknown Metal Device");
            return true;
        }
    }

    bool CreatePipeline(const char* source, NSString* entryPoint, id<MTLComputePipelineState>& pipeline)
    {
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = NO;
        id<MTLLibrary> library = [Device
            newLibraryWithSource:[NSString stringWithUTF8String:source]
            options:options error:&error];
        ReleaseObject(options);
        if (library == nil)
        {
            SetUnavailable(ErrorDescription(error, "Metal integrator compilation failed."));
            return false;
        }
        id<MTLFunction> function = [library newFunctionWithName:entryPoint];
        if (function != nil)
            pipeline = [Device newComputePipelineStateWithFunction:function error:&error];
        ReleaseObject(function);
        ReleaseObject(library);
        if (pipeline == nil)
        {
            SetUnavailable(ErrorDescription(error, "Metal integrator pipeline creation failed."));
            return false;
        }
        if (pipeline.maxTotalThreadsPerThreadgroup < ThreadsPerGroup)
        {
            SetUnavailable("The Metal device does not support the required compute group size.");
            return false;
        }
        return true;
    }

    void Shutdown()
    {
        SetUnavailable("The Metal compute backend has been shut down.");
        CachedOwner.reset();
        ReleaseFrameBuffers();
        SpectralFallback.Reset();
        ReleaseSceneBuffers();
        ReleaseObject(WhittedPipeline);
        ReleaseObject(QueryPipeline);
        ReleaseObject(PathTracingPipeline);
        ReleaseObject(DistributedRayPipeline);
        ReleaseObject(ResolvePipeline);
        ReleaseObject(SpectralResponsePipeline);
        ReleaseObject(SpectralPalettePipeline);
        SpectralResponsePipeline = SpectralPalettePipeline = nil;
        ReleaseObject(SpatialDenoisePipeline);
        SpatialDenoisePipeline = nil;
        ReleaseObject(AdaptiveDenoisePipeline);
        AdaptiveDenoisePipeline = nil;
        ReleaseObject(TemporalDenoisePipeline);
        ReleaseObject(DicingPipeline);
        TemporalDenoisePipeline = nil;
        for (auto& pipeline : GeometryGuidePipelines)
        {
            ReleaseObject(pipeline);
            pipeline = nil;
        }
        for (auto& pipeline : VcmPipelines)
        {
            ReleaseObject(pipeline);
            pipeline = nil;
        }
        ReleaseObject(BidirectionalPairs);
        ReleaseObject(BidirectionalClear);
        ReleaseObject(BidirectionalCommit);
        for (auto& pipeline : MetropolisPipelines)
        {
            ReleaseObject(pipeline);
            pipeline = nil;
        }
        for (auto& pipeline : PhotonPipelines)
        {
            ReleaseObject(pipeline);
            pipeline = nil;
        }
        for (auto& pipeline : ProgressivePipelines)
        {
            ReleaseObject(pipeline);
            pipeline = nil;
        }
        for (auto& pipeline : PointGiPipelines)
        {
            ReleaseObject(pipeline);
            pipeline = nil;
        }
        ReleaseObject(CommandQueue);
        ReleaseObject(Device);
        WhittedPipeline = nil;
        QueryPipeline = nil;
        PathTracingPipeline = nil;
        DistributedRayPipeline = nil;
        ResolvePipeline = nil;
        BidirectionalPairs = BidirectionalClear = BidirectionalCommit = nil;
        CommandQueue = nil;
        Device = nil;
    }

    [[nodiscard]] bool CommandSucceeded(
        id<MTLCommandBuffer> commandBuffer,
        const char* operation,
        std::string& error) const
    {
        if (commandBuffer == nil)
        {
            error = std::string(operation) + " did not create a command buffer.";
            return false;
        }
        if (commandBuffer.status == MTLCommandBufferStatusCompleted)
        {
            return true;
        }
        error = std::string(operation) + " failed: "
            + ErrorDescription(
                commandBuffer.error,
                "the Metal command buffer did not complete.");
        return false;
    }

    [[nodiscard]] bool CommitAndWait(
        id<MTLCommandBuffer> commandBuffer,
        const char* operation,
        const Detail::CancellationProbe& cancellation,
        std::string& error) const
    {
        if (commandBuffer == nil)
        {
            error = std::string(operation) + " did not create a command buffer.";
            return false;
        }
        const std::shared_ptr<CommandCompletionState> completion =
            std::make_shared<CommandCompletionState>();
        [commandBuffer
            addCompletedHandler:^(id<MTLCommandBuffer>)
            {
                {
                    std::lock_guard<std::mutex> lock(completion->Mutex);
                    completion->Completed = true;
                }
                completion->Condition.notify_all();
            }];
        [commandBuffer commit];

        constexpr auto CommandTimeout = std::chrono::seconds(30);
        constexpr auto CancellationPollInterval = std::chrono::milliseconds(10);
        bool cancellationObserved = false;
        const auto deadline = std::chrono::steady_clock::now() + CommandTimeout;
        std::unique_lock<std::mutex> lock(completion->Mutex);
        while (!completion->Completed)
        {
            completion->Condition.wait_for(
                lock,
                CancellationPollInterval,
                [&completion]() { return completion->Completed; });
            if (cancellation && cancellation())
            {
                cancellationObserved = true;
            }
            if (!completion->Completed
                && std::chrono::steady_clock::now() >= deadline)
            {
                lock.unlock();
                error = std::string("Terminal Metal device-lost state: ")
                    + operation
                    + " timed out after 30 seconds; the backend was disabled "
                      "to protect resources retained by the in-flight command buffer.";
                MarkTerminal(error);
                return false;
            }
        }
        lock.unlock();
        if (!CommandSucceeded(commandBuffer, operation, error))
        {
            MarkTerminal(
                std::string("Terminal Metal device-lost state: ") + error);
            return false;
        }
        if (cancellationObserved)
        {
            error = std::string(operation)
                + " completed after cancellation was requested.";
            return false;
        }
        return true;
    }

    [[nodiscard]] bool CreateBuffer(
        Buffer& output,
        std::size_t logicalSize,
        const void* data,
        const char* label,
        std::string& error) const
    {
        const std::size_t allocationSize = std::max<std::size_t>(logicalSize, 16u);
        if (allocationSize > std::numeric_limits<NSUInteger>::max())
        {
            error = std::string(label) + " exceeds Metal's addressable buffer size.";
            return false;
        }
        id<MTLBuffer> buffer = [Device
            newBufferWithLength:static_cast<NSUInteger>(allocationSize)
            options:MTLResourceStorageModeShared];
        if (buffer == nil)
        {
            error = std::string("Metal could not allocate ") + label + ".";
            return false;
        }
        buffer.label = [NSString stringWithUTF8String:label];
        if (data != nullptr && logicalSize != 0u)
        {
            void* destination = buffer.contents;
            if (destination == nullptr)
            {
                ReleaseObject(buffer);
                error = std::string("Metal could not map ") + label + ".";
                return false;
            }
            std::memcpy(destination, data, logicalSize);
        }
        output.Reset();
        output.Handle = buffer;
        output.LogicalSize = logicalSize;
        return true;
    }

    template <typename Element>
    [[nodiscard]] bool CreateVectorBuffer(
        Buffer& output,
        const std::vector<Element>& source,
        const char* label,
        std::string& error) const
    {
        if (source.size() > std::numeric_limits<std::size_t>::max() / sizeof(Element))
        {
            error = std::string(label) + " size overflows host memory.";
            return false;
        }
        return CreateBuffer(
            output,
            source.size() * sizeof(Element),
            source.empty() ? nullptr : source.data(),
            label,
            error);
    }

    void ReleaseSceneBuffers() const
    {
        InstanceRecords.Reset();
        InstanceNodes.Reset();
        InstanceError.Reset();
        InstanceNodeCount = 0u;
        Nodes.Reset();
        PrimitiveRefs.Reset();
        Triangles.Reset();
        Spheres.Reset();
        Materials.Reset();
        Lights.Reset();
        TextureDescriptors.Reset();
        TextureTexels.Reset();
        SpectralPalette.Reset();
        SpectralReferences.Reset();
        NodeCount = 0u;
        PrimitiveRefCount = 0u;
        TriangleCount = 0u;
        SphereCount = 0u;
        MaterialCount = 0u;
        LightCount = 0u;
        TextureCount = 0u;
        TextureTexelCount = 0u;
        CachedOwner.reset();
    }

    void ReleaseFrameBuffers() const
    {
        Accumulation.Reset();
        SampleCounts.Reset();
        SampleTargets.Reset();
        Output.Reset();
        GroupStats.Reset();
        FramePixelCapacity = 0u;
    }

    [[nodiscard]] bool OwnerMatches(
        const std::shared_ptr<const Scene>& owner) const
    {
        if (!owner)
        {
            return false;
        }
        const std::shared_ptr<const Scene> cached = CachedOwner.lock();
        return cached && cached.get() == owner.get();
    }

    bool EnsureSpectralPipelines(std::string& error) const
    {
        if (SpectralResponsePipeline && SpectralPalettePipeline)
            return true;
        @autoreleasepool
        {
            NSError* detail = nil;
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            options.fastMathEnabled = NO;
            id<MTLLibrary> library = [Device
                newLibraryWithSource:[NSString stringWithUTF8String:Detail::MetalSpectralPrepareSource]
                             options:options
                               error:&detail];
            ReleaseObject(options);
            if (!library)
            {
                error = ErrorDescription(detail, "Metal spectral preparation compilation failed.");
                return false;
            }
            id<MTLComputePipelineState> response = nil, palette = nil;
            const auto create = [&](NSString* name, id<MTLComputePipelineState>& pipeline) {
                id<MTLFunction> function = [library newFunctionWithName:name];
                if (function)
                    pipeline = [Device newComputePipelineStateWithFunction:function error:&detail];
                ReleaseObject(function);
                constexpr NSUInteger FitThreads = 64;
                if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < FitThreads)
                {
                    error = ErrorDescription(
                        detail, "Metal spectral preparation cannot create its threadgroup pipeline.");
                    return false;
                }
                return true;
            };
            const bool ready =
                create(@"SpectralPrepareResponse", response) && create(@"SpectralPreparePalette", palette);
            ReleaseObject(library);
            if (!ready)
            {
                ReleaseObject(response);
                ReleaseObject(palette);
                return false;
            }
            ReleaseObject(SpectralResponsePipeline);
            ReleaseObject(SpectralPalettePipeline);
            SpectralResponsePipeline = response;
            SpectralPalettePipeline = palette;
            return true;
        }
    }

    bool PrepareSpectralCoefficients(const SpectralSceneData::Request* sources, uint32_t count,
                                     const Detail::CancellationProbe& cancellation, Buffer& output,
                                     std::string& error) const
    {
        namespace S = SpectralSceneData;
        constexpr uint32_t ResponseCount = 256, FitThreads = 64, BatchSize = 8192, FitConverged = 0;
        if (!count || !EnsureSpectralPipelines(error))
        {
            if (!count)
                error = "Metal spectral preparation needs a nonempty palette.";
            return false;
        }
        const size_t inputBytes = size_t(count) * sizeof(S::Request);
        const size_t outputBytes = size_t(count) * sizeof(S::GpuSpectrum);
        if (std::max(inputBytes, outputBytes) > Device.maxBufferLength)
        {
            error = "Metal spectral palette exceeds the device buffer limit.";
            return false;
        }
        Buffer input, response, coefficients;
        if (!CreateBuffer(input, inputBytes, sources, "Arvo Metal spectral source colors", error) ||
            !CreateBuffer(response, ResponseCount * sizeof(Vec4), nullptr, "Arvo Metal spectral quadrature",
                          error) ||
            !CreateBuffer(coefficients, outputBytes, nullptr, "Arvo Metal spectral coefficients", error))
            return false;
        if (cancellation())
        {
            error = "Metal spectral preparation cancelled.";
            return false;
        }
        @autoreleasepool
        {
            id<MTLCommandBuffer> command = [CommandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!command || !encoder)
            {
                error = "Metal spectral quadrature encoder is unavailable.";
                return false;
            }
            [encoder setComputePipelineState:SpectralResponsePipeline];
            [encoder setBuffer:response.Handle offset:0 atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake((ResponseCount + FitThreads - 1) / FitThreads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(FitThreads, 1, 1)];
            [encoder endEncoding];
            if (!CommitAndWait(command, "Metal spectral quadrature", cancellation, error))
                return false;
        }
        for (uint32_t first = 0; first < count;)
        {
            if (cancellation())
            {
                error = "Metal spectral fitting cancelled.";
                return false;
            }
            const uint32_t batch = std::min(BatchSize, count - first);
            @autoreleasepool
            {
                id<MTLCommandBuffer> command = [CommandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (!command || !encoder)
                {
                    error = "Metal spectral fit encoder is unavailable.";
                    return false;
                }
                [encoder setComputePipelineState:SpectralPalettePipeline];
                [encoder setBuffer:input.Handle offset:size_t(first) * sizeof(S::Request) atIndex:0];
                [encoder setBuffer:response.Handle offset:0 atIndex:1];
                [encoder setBuffer:coefficients.Handle
                            offset:size_t(first) * sizeof(S::GpuSpectrum)
                           atIndex:2];
                [encoder setBytes:&batch length:sizeof(batch) atIndex:3];
                [encoder dispatchThreadgroups:MTLSizeMake((batch + FitThreads - 1) / FitThreads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(FitThreads, 1, 1)];
                [encoder endEncoding];
                if (!CommitAndWait(command, "Metal spectral palette fitting", cancellation, error))
                    return false;
            }
            const auto* data = static_cast<const S::GpuSpectrum*>(coefficients.Handle.contents);
            if (!data)
            {
                error = "Metal spectral status is not CPU-visible.";
                return false;
            }
            for (uint32_t index = first; index < first + batch; ++index)
            {
                const auto& fit = data[index];
                if (fit.Status != FitConverged || !std::isfinite(fit.Quadratic) ||
                    !std::isfinite(fit.Linear) || !std::isfinite(fit.Offset) || !std::isfinite(fit.Scale) ||
                    !std::isfinite(fit.Constant) || !std::isfinite(fit.Error) || fit.Scale < 0 ||
                    fit.Constant < 0 || fit.Error < 0)
                {
                    error = "Metal spectral fit failed at entry " + std::to_string(index);
                    return false;
                }
            }
            first += batch;
        }
        if (cancellation())
        {
            error = "Metal spectral coefficient publication cancelled.";
            return false;
        }
        output = std::move(coefficients);
        return true;
    }

    bool UpdateSpectralMedium(const MediumOptions& medium, const Detail::CancellationProbe& cancellation,
                              std::string& error) const
    {
        if (SpectralMedium == medium)
            return true;
        namespace S = SpectralSceneData;
        const S::Request sources[]{{medium.Enabled ? medium.Absorption : Float3(), S::Role::Nonnegative},
                                   {medium.Enabled ? medium.Scattering : Float3(), S::Role::Nonnegative}};
        Buffer candidate;
        if (!PrepareSpectralCoefficients(sources, uint32_t(std::size(sources)), cancellation, candidate,
                                         error))
            return false;
        if (cancellation())
        {
            error = "Metal spectral medium publication cancelled.";
            return false;
        }
        auto* destination = static_cast<S::GpuSpectrum*>(SpectralPalette.Handle.contents);
        const void* source = candidate.Handle.contents;
        if (!destination || !source)
        {
            error = "Metal spectral medium buffers are not CPU-visible.";
            return false;
        }
        // The queue is serialized and the fit command has completed. Shared
        // storage needs no managed-memory synchronization for this small update.
        std::memcpy(destination + SpectralSceneSourceCount, source, candidate.LogicalSize);
        SpectralMedium = medium;
        return true;
    }

    [[nodiscard]] bool EnsureScene(const Scene& scene, const std::shared_ptr<const Scene>& owner,
                                   AccelerationOptions acceleration,
                                   const Detail::CancellationProbe& cancellation, std::string& error,
                                   const MediumOptions* spectralMedium = nullptr) const
    {
        if (OwnerMatches(owner) && CachedAcceleration == acceleration &&
            (!spectralMedium || SpectralPalette.Handle != nil))
        {
            return !spectralMedium || UpdateSpectralMedium(*spectralMedium, cancellation, error);
        }
        PackedScene packed;
        if (!MetalScene::Pack(scene, acceleration, cancellation, packed, error))
        {
            return false;
        }
        SpectralSceneData::GpuScene spectral;
        Buffer spectralPalette, spectralReferences;
        if (spectralMedium)
        {
            if (MetalSpectralHost::BuildGpuInputs(scene, *spectralMedium, cancellation, spectral, error) !=
                RenderStatus::Completed)
                return false;
            for (auto& triangle : packed.Triangles)
            {
                const auto& a = triangle.Color0;
                const auto& b = triangle.Color1;
                const auto& c = triangle.Color2;
                const auto link =
                    spectral.TriangleReferences.find({a.X, a.Y, a.Z, b.X, b.Y, b.Z, c.X, c.Y, c.Z});
                if (link == spectral.TriangleReferences.end())
                {
                    error = "Metal packed vertex colors do not match the spectral scene.";
                    return false;
                }
                triangle.Padding0 = link->second;
            }
            if (!PrepareSpectralCoefficients(spectral.Sources.data(), uint32_t(spectral.Sources.size()),
                                             cancellation, spectralPalette, error) ||
                !CreateVectorBuffer(spectralReferences, spectral.References, "Arvo Metal spectral references",
                                    error))
                return false;
        }
        if (cancellation())
        {
            error = "Metal scene construction was cancelled.";
            return false;
        }

        Buffer nodes;
        Buffer primitiveRefs;
        Buffer triangles;
        Buffer spheres;
        Buffer materials;
        Buffer lights;
        Buffer textureDescriptors;
        Buffer textureTexels;
        Buffer instanceRecords, instanceNodes, instanceError;
        if (!CreateVectorBuffer(nodes, packed.Nodes, "Arvo Metal BVH nodes", error)
            || !CreateVectorBuffer(
                primitiveRefs,
                packed.PrimitiveRefs,
                "Arvo Metal primitive references",
                error)
            || !CreateVectorBuffer(
                triangles,
                packed.Triangles,
                "Arvo Metal triangles",
                error)
            || !CreateVectorBuffer(
                spheres,
                packed.Spheres,
                "Arvo Metal spheres",
                error)
            || !CreateVectorBuffer(
                materials,
                packed.Materials,
                "Arvo Metal materials",
                error)
            || !CreateVectorBuffer(
                lights,
                packed.Lights,
                "Arvo Metal lights",
                error)
            || !CreateVectorBuffer(
                textureDescriptors,
                packed.TextureAtlas.Descriptors,
                "Arvo Metal texture descriptors",
                error)
            || !CreateVectorBuffer(
                textureTexels,
                packed.TextureAtlas.Texels,
                "Arvo Metal texture texels",
                error)
            || !CreateVectorBuffer(instanceRecords, packed.Instances, "Arvo Metal instance records", error)
            || !CreateVectorBuffer(instanceNodes, packed.InstanceNodes, "Arvo Metal instance TLAS", error)
            || !CreateBuffer(instanceError, sizeof(uint32_t), nullptr, "Arvo Metal traversal error", error))
        {
            return false;
        }
        if (cancellation())
        {
            error = "Metal scene upload was cancelled.";
            return false;
        }

        ReleaseSceneBuffers();
        InstanceRecords = std::move(instanceRecords);
        InstanceNodes = std::move(instanceNodes);
        InstanceError = std::move(instanceError);
        InstanceNodeCount = uint32_t(packed.InstanceNodes.size());
        Nodes = std::move(nodes);
        PrimitiveRefs = std::move(primitiveRefs);
        Triangles = std::move(triangles);
        Spheres = std::move(spheres);
        Materials = std::move(materials);
        Lights = std::move(lights);
        TextureDescriptors = std::move(textureDescriptors);
        TextureTexels = std::move(textureTexels);
        NodeCount = static_cast<std::uint32_t>(packed.Nodes.size());
        PrimitiveRefCount =
            static_cast<std::uint32_t>(packed.PrimitiveRefs.size());
        TriangleCount = static_cast<std::uint32_t>(packed.Triangles.size());
        SphereCount = static_cast<std::uint32_t>(packed.Spheres.size());
        MaterialCount = static_cast<std::uint32_t>(packed.Materials.size());
        LightCount = static_cast<std::uint32_t>(packed.Lights.size());
        TextureCount = static_cast<std::uint32_t>(
            packed.TextureAtlas.Descriptors.size());
        TextureTexelCount = static_cast<std::uint32_t>(
            packed.TextureAtlas.Texels.size());
        CachedOwner = owner;
        CachedAcceleration = acceleration;
        SpectralPalette = std::move(spectralPalette);
        SpectralReferences = std::move(spectralReferences);
        SpectralSceneSourceCount = spectral.SceneSourceCount;
        if (spectralMedium)
            SpectralMedium = *spectralMedium;
        return true;
    }

    [[nodiscard]] bool EnsureFrameBuffers(
        std::uint32_t pixelCount,
        std::string& error) const
    {
        if (!SpectralFallback.Handle)
        {
            const SpectralSceneData::GpuSpectrum zero{};
            if (!CreateBuffer(SpectralFallback, sizeof(zero), &zero, "Arvo Metal RGB spectral fallback",
                              error))
                return false;
        }
        if (FramePixelCapacity >= pixelCount)
        {
            return true;
        }
        if (pixelCount > std::numeric_limits<std::size_t>::max() / sizeof(Vec4))
        {
            error = "Metal frame buffer dimensions overflow host memory.";
            return false;
        }

        Buffer accumulation;
        Buffer sampleCounts;
        Buffer sampleTargets;
        Buffer output;
        Buffer groupStats;
        const std::size_t groupCapacity =
            (PixelChunkSize + ThreadsPerGroup - 1u) / ThreadsPerGroup;
        if (!CreateBuffer(
                accumulation,
                static_cast<std::size_t>(pixelCount) * sizeof(Vec4),
                nullptr,
                "Arvo Metal accumulation",
                error)
            || !CreateBuffer(
                sampleCounts,
                static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t),
                nullptr,
                "Arvo Metal sample counts",
                error)
            || !CreateBuffer(
                sampleTargets,
                static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t),
                nullptr,
                "Arvo Metal sample targets",
                error)
            || !CreateBuffer(
                output,
                static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t),
                nullptr,
                "Arvo Metal output",
                error)
            || !CreateBuffer(
                groupStats,
                groupCapacity * sizeof(GroupStatistics),
                nullptr,
                "Arvo Metal group statistics",
                error))
        {
            return false;
        }
        ReleaseFrameBuffers();
        Accumulation = std::move(accumulation);
        SampleCounts = std::move(sampleCounts);
        SampleTargets = std::move(sampleTargets);
        Output = std::move(output);
        GroupStats = std::move(groupStats);
        FramePixelCapacity = pixelCount;
        return true;
    }

    [[nodiscard]] bool ClearFrame(
        std::uint32_t pixelCount,
        const Detail::CancellationProbe& cancellation,
        std::string& error) const
    {
        void* accumulation = Accumulation.Handle.contents;
        void* sampleCounts = SampleCounts.Handle.contents;
        void* sampleTargets = SampleTargets.Handle.contents;
        if (accumulation == nullptr || sampleCounts == nullptr || sampleTargets == nullptr)
        {
            error = "Metal frame buffers are not CPU-visible.";
            return false;
        }
        constexpr std::size_t ClearChunkSize = 32u * 1024u * 1024u;
        const std::size_t accumulationBytes =
            static_cast<std::size_t>(pixelCount) * sizeof(Vec4);
        const std::size_t sampleBytes =
            static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t);
        for (std::size_t offset = 0u;
             offset < accumulationBytes;
             offset += ClearChunkSize)
        {
            if (cancellation())
            {
                error = "Metal frame clear was cancelled.";
                return false;
            }
            std::memset(
                static_cast<std::uint8_t*>(accumulation) + offset,
                0,
                std::min(ClearChunkSize, accumulationBytes - offset));
        }
        for (std::size_t offset = 0u; offset < sampleBytes; offset += ClearChunkSize)
        {
            if (cancellation())
            {
                error = "Metal frame clear was cancelled.";
                return false;
            }
            std::memset(
                static_cast<std::uint8_t*>(sampleCounts) + offset,
                0,
                std::min(ClearChunkSize, sampleBytes - offset));
            std::memset(
                static_cast<std::uint8_t*>(sampleTargets) + offset,
                0,
                std::min(ClearChunkSize, sampleBytes - offset));
        }
        return true;
    }

    bool ResetTraversalError(std::string& error) const
    {
        void* value = InstanceError.Handle.contents;
        if (!value)
        {
            error = "Metal traversal error buffer is not CPU-visible.";
            return false;
        }
        std::memset(value, 0, sizeof(uint32_t));
        return true;
    }

    bool CheckTraversalError(std::string& error) const
    {
        uint32_t value = 0;
        std::memcpy(&value, InstanceError.Handle.contents, sizeof(value));
        if (value == 0) return true;
        error = "Metal affine ray or normal exceeds native primitive arithmetic range.";
        return false;
    }

    void BindInstanceBuffers(id<MTLComputeCommandEncoder> encoder) const
    {
        [encoder setBuffer:InstanceRecords.Handle offset:0u atIndex:InstanceRecordBinding];
        [encoder setBuffer:InstanceNodes.Handle offset:0u atIndex:InstanceNodeBinding];
        [encoder setBuffer:InstanceError.Handle offset:0u atIndex:InstanceErrorBinding];
    }

    [[nodiscard]] bool DispatchTrace(const FrameConstants& constants, const MediumData& medium,
                                     const MotionData::FrameExposure& cameraExposure, GroupStatistics& totals,
                                     const Detail::CancellationProbe& cancellation, std::string& error) const
    {
        if (!ResetTraversalError(error)) return false;
        @autoreleasepool
        {
            id<MTLCommandBuffer> commandBuffer = [CommandQueue commandBuffer];
            if (commandBuffer == nil)
            {
                error = "Metal trace dispatch did not create a command buffer.";
                return false;
            }
            id<MTLComputeCommandEncoder> encoder =
                [commandBuffer computeCommandEncoder];
            if (encoder == nil)
            {
                error = "Metal trace dispatch did not create a compute encoder.";
                return false;
            }
            id<MTLComputePipelineState> pipeline = nil;
            switch (static_cast<Algorithm>(constants.Integrator))
            {
            case Algorithm::RayTracing: pipeline = WhittedPipeline; break;
            case Algorithm::PathTracing: pipeline = PathTracingPipeline; break;
            case Algorithm::DistributedRayTracing: pipeline = DistributedRayPipeline; break;
            default:
                [encoder endEncoding];
                error = "No native Metal pipeline for the integrator";
                return false;
            }
            [encoder setComputePipelineState:pipeline];
            if (constants.Integrator == uint32_t(Algorithm::PathTracing))
            {
                constexpr NSUInteger MediumBinding = 16;
                [encoder setBytes:&medium length:sizeof(medium) atIndex:MediumBinding];
                constexpr NSUInteger CameraExposureBinding = 19;
                [encoder setBytes:&cameraExposure
                           length:sizeof(cameraExposure)
                          atIndex:CameraExposureBinding];
                constexpr NSUInteger SpectralPaletteBinding = 17, SpectralReferenceBinding = 18;
                [encoder setBuffer:SpectralPalette.Handle ? SpectralPalette.Handle : SpectralFallback.Handle
                            offset:0
                           atIndex:SpectralPaletteBinding];
                [encoder
                    setBuffer:SpectralReferences.Handle ? SpectralReferences.Handle : SpectralFallback.Handle
                       offset:0
                      atIndex:SpectralReferenceBinding];
            }
            if (constants.Integrator == uint32_t(Algorithm::DistributedRayTracing))
            {
                constexpr NSUInteger CameraExposureBinding = 19;
                [encoder setBytes:&cameraExposure length:sizeof(cameraExposure) atIndex:CameraExposureBinding];
            }
            [encoder setBuffer:Nodes.Handle offset:0u atIndex:0u];
            [encoder setBuffer:PrimitiveRefs.Handle offset:0u atIndex:1u];
            [encoder setBuffer:Triangles.Handle offset:0u atIndex:2u];
            [encoder setBuffer:Spheres.Handle offset:0u atIndex:3u];
            [encoder setBuffer:Materials.Handle offset:0u atIndex:4u];
            [encoder setBuffer:Lights.Handle offset:0u atIndex:5u];
            [encoder
                setBytes:&constants
                length:sizeof(constants)
                atIndex:6u];
            [encoder setBuffer:Accumulation.Handle offset:0u atIndex:7u];
            [encoder setBuffer:SampleCounts.Handle offset:0u atIndex:8u];
            [encoder setBuffer:GroupStats.Handle offset:0u atIndex:9u];
            [encoder setBuffer:TextureDescriptors.Handle offset:0u atIndex:10u];
            [encoder setBuffer:TextureTexels.Handle offset:0u atIndex:11u];
            [encoder setBuffer:SampleTargets.Handle offset:0u atIndex:12u];
            BindInstanceBuffers(encoder);
            [encoder
                dispatchThreadgroups:MTLSizeMake(constants.GroupCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(ThreadsPerGroup, 1u, 1u)];
            [encoder endEncoding];
            if (!CommitAndWait(
                    commandBuffer,
                    "Metal trace dispatch",
                    cancellation,
                    error))
            {
                return false;
            }
            if (!CheckTraversalError(error)) return false;
            if (cancellation())
            {
                error = "Metal trace dispatch was cancelled.";
                return false;
            }
        }

        const auto* groups =
            static_cast<const GroupStatistics*>(GroupStats.Handle.contents);
        if (groups == nullptr)
        {
            error = "Metal group statistics are not CPU-visible.";
            return false;
        }
        for (std::uint32_t group = 0u; group < constants.GroupCount; ++group)
        {
            const GroupStatistics& value = groups[group];
            auto add = [](Counter64Parts& destination, std::uint64_t amount)
            {
                const std::uint64_t total = destination.Value() + amount;
                destination.Low = static_cast<std::uint32_t>(total);
                destination.High = static_cast<std::uint32_t>(total >> 32u);
            };
            add(totals.PrimaryRays, value.PrimaryRays.Value());
            add(totals.ShadowRays, value.ShadowRays.Value());
            add(totals.ReflectionRays, value.ReflectionRays.Value());
            add(totals.PrimitiveTests, value.PrimitiveTests.Value());
        }
        return true;
    }

    [[nodiscard]] bool ResolvePixels(
        FrameConstants constants,
        std::uint32_t pixelCount,
        const Detail::CancellationProbe& cancellation,
        std::string& error) const
    {
        for (std::uint32_t begin = 0u; begin < pixelCount; begin += PixelChunkSize)
        {
            if (cancellation())
            {
                error = "Metal resolve was cancelled.";
                return false;
            }
            constants.PixelBegin = begin;
            constants.PixelCount = std::min(PixelChunkSize, pixelCount - begin);
            const std::uint32_t groupCount =
                (constants.PixelCount + ThreadsPerGroup - 1u) / ThreadsPerGroup;
            @autoreleasepool
            {
                id<MTLCommandBuffer> commandBuffer = [CommandQueue commandBuffer];
                if (commandBuffer == nil)
                {
                    error = "Metal resolve did not create a command buffer.";
                    return false;
                }
                id<MTLComputeCommandEncoder> encoder =
                    [commandBuffer computeCommandEncoder];
                if (encoder == nil)
                {
                    error = "Metal resolve did not create a compute encoder.";
                    return false;
                }
                [encoder setComputePipelineState:ResolvePipeline];
                [encoder setBuffer:Accumulation.Handle offset:0u atIndex:0u];
                [encoder setBuffer:SampleCounts.Handle offset:0u atIndex:1u];
                [encoder setBuffer:Output.Handle offset:0u atIndex:2u];
                [encoder
                    setBytes:&constants
                    length:sizeof(constants)
                    atIndex:3u];
                [encoder
                    dispatchThreadgroups:MTLSizeMake(groupCount, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(ThreadsPerGroup, 1u, 1u)];
                [encoder endEncoding];
                if (!CommitAndWait(
                        commandBuffer,
                        "Metal resolve",
                        cancellation,
                        error))
                {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] RayQueryResult QueryNative(const Scene& scene, const std::vector<Ray>& rays, AccelerationOptions acceleration,
        const Detail::CancellationProbe& cancellation) const
    {
        RayQueryResult result;
        if (!ValidateAcceleration(acceleration, result.Error)
                || !ValidateRayQueries(rays, result.Error)
            || !ValidateInput(scene, Camera{}, RenderSettings{}, result.Error, false)) return result;
        if (cancellation()) { result.Status = RenderStatus::Cancelled; return result; }
        if (rays.empty()) { result.Status = RenderStatus::Completed; return result; }
        if (!EnsureScene(scene, {}, acceleration, cancellation, result.Error))
        {
            if (cancellation()) result.Status = RenderStatus::Cancelled;
            return result;
        }
        std::vector<QueryRay> input;
        input.reserve(rays.size());
        for (const Ray& ray : rays)
            input.push_back({{ray.Origin.X, ray.Origin.Y, ray.Origin.Z, ray.MinDistance},
                {ray.Direction.X, ray.Direction.Y, ray.Direction.Z, ray.MaxDistance}});
        Buffer queryInput, queryOutput;
        if (!CreateVectorBuffer(queryInput, input, "Arvo Metal query rays", result.Error)
            || !CreateBuffer(queryOutput, input.size() * sizeof(RayQueryHit), nullptr,
                "Arvo Metal query hits", result.Error)) return result;
        FrameConstants constants;
        constants.Acceleration = static_cast<std::uint32_t>(acceleration.Type);
        constants.NodeCount = NodeCount;
        constants.InstanceNodeCount = InstanceNodeCount;
        constants.PrimitiveRefCount = PrimitiveRefCount;
        constants.TriangleCount = TriangleCount;
        constants.SphereCount = SphereCount;
        for (std::uint32_t begin = 0; begin < input.size(); begin += PixelChunkSize)
        {
            if (cancellation()) { result.Status = RenderStatus::Cancelled; return result; }
            constants.PixelBegin = begin;
            constants.PixelCount = std::min(PixelChunkSize, static_cast<std::uint32_t>(input.size()) - begin);
            if (!ResetTraversalError(result.Error)) return result;
            @autoreleasepool
            {
                id<MTLCommandBuffer> commandBuffer = [CommandQueue commandBuffer];
                if (commandBuffer == nil) { result.Error = "Metal query did not create a command buffer."; return result; }
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (encoder == nil) { result.Error = "Metal query did not create a compute encoder."; return result; }
                [encoder setComputePipelineState:QueryPipeline];
                [encoder setBuffer:Nodes.Handle offset:0u atIndex:0u];
                [encoder setBuffer:PrimitiveRefs.Handle offset:0u atIndex:1u];
                [encoder setBuffer:Triangles.Handle offset:0u atIndex:2u];
                [encoder setBuffer:Spheres.Handle offset:0u atIndex:3u];
                [encoder setBytes:&constants length:sizeof(constants) atIndex:4u];
                [encoder setBuffer:queryInput.Handle offset:0u atIndex:5u];
                [encoder setBuffer:queryOutput.Handle offset:0u atIndex:6u];
                BindInstanceBuffers(encoder);
                const std::uint32_t groups = (constants.PixelCount + ThreadsPerGroup - 1u) / ThreadsPerGroup;
                [encoder dispatchThreadgroups:MTLSizeMake(groups, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(ThreadsPerGroup, 1u, 1u)];
                [encoder endEncoding];
                if (!CommitAndWait(commandBuffer, "Metal ray queries", cancellation, result.Error))
                {
                    if (cancellation()) result.Status = RenderStatus::Cancelled;
                    return result;
                }
                if (!CheckTraversalError(result.Error)) return result;
            }
        }
        const void* hits = queryOutput.Handle.contents;
        if (hits == nullptr) { result.Error = "Metal query hits are not CPU-visible."; return result; }
        result.Hits.resize(rays.size());
        constexpr size_t ReadbackChunkSize = 32u * 1024u * 1024u;
        const size_t bytes = result.Hits.size() * sizeof(RayQueryHit);
        for (size_t offset = 0; offset < bytes; offset += ReadbackChunkSize)
        {
            if (cancellation()) { result.Status = RenderStatus::Cancelled; return result; }
            std::memcpy(reinterpret_cast<std::uint8_t*>(result.Hits.data()) + offset,
                static_cast<const std::uint8_t*>(hits) + offset, std::min(ReadbackChunkSize, bytes - offset));
        }
        result.Status = cancellation() ? RenderStatus::Cancelled : RenderStatus::Completed;
        return result;
    }

    bool EnsureBidirectionalPipelines(std::string& error) const
    {
        if (BidirectionalPairs && BidirectionalClear && BidirectionalCommit)
            return true;
        @autoreleasepool
        {
            NSError* detail = nil;
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            options.fastMathEnabled = NO;
            id<MTLLibrary> library =
                [Device newLibraryWithSource:[NSString stringWithUTF8String:Detail::MetalBidirectionalSource]
                                     options:options
                                       error:&detail];
            ReleaseObject(options);
            if (!library)
            {
                error = ErrorDescription(detail, "Metal BDPT compilation failed.");
                return false;
            }
            id<MTLComputePipelineState> pairs = nil, clear = nil, commit = nil;
            auto create = [&](NSString* name, id<MTLComputePipelineState>& pipeline) {
                id<MTLFunction> function = [library newFunctionWithName:name];
                if (function)
                    pipeline = [Device newComputePipelineStateWithFunction:function error:&detail];
                ReleaseObject(function);
                if (!pipeline)
                {
                    error = ErrorDescription(detail, "Metal BDPT pipeline creation failed.");
                    return false;
                }
                if (pipeline.maxTotalThreadsPerThreadgroup < MetalBdpt::ThreadsPerGroup)
                {
                    error = "Metal BDPT pipeline cannot support its requested threadgroup size.";
                    return false;
                }
                return true;
            };
            const bool ready = create(@"ArvoBidirectionalPairs", pairs) &&
                               create(@"ArvoBidirectionalClear", clear) &&
                               create(@"ArvoBidirectionalCommit", commit);
            ReleaseObject(library);
            if (!ready)
            {
                ReleaseObject(pairs);
                ReleaseObject(clear);
                ReleaseObject(commit);
                return false;
            }
            // The serialized Metal queue publishes all three pipelines together.
            BidirectionalPairs = pairs;
            BidirectionalClear = clear;
            BidirectionalCommit = commit;
            return true;
        }
    }

    bool RenderBidirectional(const Scene& scene, FrameConstants constants,
                             const Detail::CancellationProbe& cancellation, RenderResult& result) const
    {
        if (!EnsureBidirectionalPipelines(result.Error))
            return false;
        const auto emission = MetalEmission::PrepareEmitters(scene);
        for (const auto& source : emission.Sources)
            if (!(source.Flags & EmitterCastsShadows))
            {
                result.Error = "Metal BDPT requires shadow casting on every emitter.";
                return false;
            }
        const uint32_t pixels = constants.Width * constants.Height;
        const uint32_t fullPixels = constants.FullWidth * constants.FullHeight;
        const uint32_t capacity = std::min(MetalBdpt::BatchCapacity, fullPixels);
        const size_t vertexBytes = size_t(capacity) * MetalBdpt::PairSubpaths * MetalBdpt::MaximumVertices *
                                   sizeof(MetalBdpt::Vertex);
        Buffer emitters, vertices, pass, statistics, failures;
        if (!CreateVectorBuffer(emitters, emission.Sources, "Arvo Metal BDPT sources", result.Error) ||
            !CreateBuffer(vertices, vertexBytes, nullptr, "Arvo Metal BDPT vertices", result.Error) ||
            !CreateBuffer(pass, size_t(pixels) * MetalBdpt::PassPixelBytes, nullptr, "Arvo Metal BDPT pass",
                          result.Error) ||
            !CreateBuffer(statistics, size_t(capacity) * sizeof(GroupStatistics), nullptr,
                          "Arvo Metal BDPT counters", result.Error) ||
            !CreateBuffer(failures, sizeof(uint32_t), nullptr, "Arvo Metal BDPT numerical status",
                          result.Error))
            return false;
        MetalBdpt::Controls controls{{emission.Center.X, emission.Center.Y, emission.Center.Z},
                                     emission.Radius,
                                     uint32_t(emission.Sources.size()),
                                     0,
                                     0,
                                     0};
        auto dispatch = [&](id<MTLComputePipelineState> pipeline, uint32_t threads) {
            if (cancellation() || !ResetTraversalError(result.Error))
                return false;
            @autoreleasepool
            {
                // Default command buffers retain every bound resource, including
                // these frame-local buffers, until asynchronous completion.
                id<MTLCommandBuffer> command = [CommandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (!command || !encoder)
                {
                    result.Error = "Metal BDPT could not create its command buffer or encoder.";
                    return false;
                }
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:Nodes.Handle offset:0 atIndex:0];
                [encoder setBuffer:PrimitiveRefs.Handle offset:0 atIndex:1];
                [encoder setBuffer:Triangles.Handle offset:0 atIndex:2];
                [encoder setBuffer:Spheres.Handle offset:0 atIndex:3];
                [encoder setBuffer:Materials.Handle offset:0 atIndex:4];
                [encoder setBuffer:emitters.Handle offset:0 atIndex:5];
                [encoder setBytes:&constants length:sizeof(constants) atIndex:6];
                [encoder setBuffer:Accumulation.Handle offset:0 atIndex:7];
                [encoder setBuffer:SampleCounts.Handle offset:0 atIndex:8];
                [encoder setBuffer:statistics.Handle offset:0 atIndex:9];
                [encoder setBuffer:TextureDescriptors.Handle offset:0 atIndex:10];
                [encoder setBuffer:TextureTexels.Handle offset:0 atIndex:11];
                [encoder setBuffer:SampleTargets.Handle offset:0 atIndex:12];
                BindInstanceBuffers(encoder);
                [encoder setBuffer:vertices.Handle offset:0 atIndex:MetalBdpt::VertexBinding];
                [encoder setBuffer:pass.Handle offset:0 atIndex:MetalBdpt::PassBinding];
                [encoder setBytes:&controls length:sizeof(controls) atIndex:MetalBdpt::ControlsBinding];
                [encoder setBuffer:failures.Handle offset:0 atIndex:MetalBdpt::ErrorBinding];
                const uint32_t groups =
                    (threads + MetalBdpt::ThreadsPerGroup - 1) / MetalBdpt::ThreadsPerGroup;
                [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(MetalBdpt::ThreadsPerGroup, 1, 1)];
                [encoder endEncoding];
                if (!CommitAndWait(command, "Metal BDPT dispatch", cancellation, result.Error) ||
                    !CheckTraversalError(result.Error))
                    return false;
            }
            const void* status = failures.Handle.contents;
            if (!status)
            {
                result.Error = "Metal BDPT numerical status is not CPU-visible.";
                return false;
            }
            uint32_t flags;
            std::memcpy(&flags, status, sizeof(flags));
            if (flags)
            {
                result.Error = "Metal BDPT encountered nonfinite path or film arithmetic.";
                return false;
            }
            return true;
        };
        // Light paths cover the full image even for a cropped output. Complete
        // passes preserve incoming splats and Welford's observation count.
        for (uint32_t sample = 0; sample < constants.TotalSamples; ++sample)
        {
            constants.SampleBegin = sample;
            if (!dispatch(BidirectionalClear, pixels))
                return false;
            for (uint32_t begin = 0; begin < fullPixels; begin += capacity)
            {
                controls.FirstPath = begin;
                controls.PathCount = std::min(capacity, fullPixels - begin);
                if (!dispatch(BidirectionalPairs, controls.PathCount))
                    return false;
                const auto* rows = static_cast<const GroupStatistics*>(statistics.Handle.contents);
                if (!rows)
                {
                    result.Error = "Metal BDPT statistics are not CPU-visible.";
                    return false;
                }
                for (uint32_t i = 0; i < controls.PathCount; ++i)
                {
                    result.Statistics.PrimaryRays += rows[i].PrimaryRays.Value();
                    result.Statistics.ShadowRays += rows[i].ShadowRays.Value();
                    result.Statistics.ReflectionRays += rows[i].ReflectionRays.Value();
                    result.Statistics.PrimitiveTests += rows[i].PrimitiveTests.Value();
                }
            }
            if (!dispatch(BidirectionalCommit, pixels))
                return false;
        }
        return true;
    }

    bool EnsureMetropolisPipelines(std::string& error) const
    {
        if (MetropolisPipelines[MetalMlt::BootstrapStage])
            return true;
        @autoreleasepool
        {
            NSError* detail = nil;
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            // WideMath's residuals require the exact arithmetic expression order.
            options.fastMathEnabled = NO;
            if (@available(macOS 11.0, *))
                options.languageVersion = MTLLanguageVersion2_3;
            else
            {
                ReleaseObject(options);
                error = "Metal MLT requires macOS 11 or later for MSL 2.3 persistent 64-bit random state.";
                return false;
            }
            id<MTLLibrary> library =
                [Device newLibraryWithSource:[NSString stringWithUTF8String:Detail::MetalMetropolisSource]
                                     options:options
                                       error:&detail];
            ReleaseObject(options);
            if (!library)
            {
                error = ErrorDescription(detail, "Metal MLT compilation failed.");
                return false;
            }
            NSString* entries[MetalMlt::StageCount] = {
                @"ArvoMetropolisBootstrap", @"ArvoMetropolisDistribution", @"ArvoMetropolisInitialize",
                @"ArvoMetropolisMutate",    @"ArvoMetropolisClear",        @"ArvoMetropolisCommit"};
            id<MTLComputePipelineState> compiled[MetalMlt::StageCount] = {};
            bool ready = true;
            for (uint32_t stage = 0; stage < MetalMlt::StageCount && ready; ++stage)
            {
                id<MTLFunction> entry = [library newFunctionWithName:entries[stage]];
                if (entry)
                    compiled[stage] = [Device newComputePipelineStateWithFunction:entry error:&detail];
                ReleaseObject(entry);
                if (!compiled[stage])
                {
                    error = ErrorDescription(detail, "Metal MLT pipeline creation failed.");
                    ready = false;
                }
                else if (compiled[stage].maxTotalThreadsPerThreadgroup < MetalMlt::ThreadsPerGroup)
                {
                    error = "Metal MLT pipeline cannot support its requested threadgroup size.";
                    ready = false;
                }
            }
            ReleaseObject(library);
            if (!ready)
            {
                for (auto pipeline : compiled)
                    ReleaseObject(pipeline);
                return false;
            }
            // Publish only a complete set on the serialized RHI queue.
            for (uint32_t stage = 0; stage < MetalMlt::StageCount; ++stage)
                MetropolisPipelines[stage] = compiled[stage];
            return true;
        }
    }

    bool RenderMetropolis(const Scene& scene, FrameConstants constants, const MetropolisOptions& options,
                          const Detail::CancellationProbe& cancellation, RenderResult& result) const
    {
        if (!EnsureMetropolisPipelines(result.Error))
            return false;
        const auto emission = MetalEmission::PrepareEmitters(scene);
        for (const auto& source : emission.Sources)
            if (!(source.Flags & EmitterCastsShadows))
            {
                result.Error = "Metal MLT requires shadow casting on every emitter.";
                return false;
            }
        const uint32_t pixels = constants.Width * constants.Height,
                       fullPixels = constants.FullWidth * constants.FullHeight;
        const uint32_t chainCount = std::min(options.ChainCount, fullPixels);
        Buffer emitters, chains, bootstrap, normalization, pass, statistics, failures;
        if (!CreateVectorBuffer(emitters, emission.Sources, "Arvo Metal MLT emitters", result.Error) ||
            !CreateBuffer(chains, size_t(chainCount) * sizeof(MetalMlt::Chain), nullptr,
                          "Arvo Metal MLT chains", result.Error) ||
            !CreateBuffer(bootstrap, size_t(options.BootstrapSamples) * sizeof(MetalMlt::Bootstrap), nullptr,
                          "Arvo Metal MLT bootstrap", result.Error) ||
            !CreateBuffer(normalization, sizeof(MetalMlt::Distribution), nullptr,
                          "Arvo Metal MLT normalization", result.Error) ||
            !CreateBuffer(pass, size_t(pixels) * sizeof(Vec4), nullptr, "Arvo Metal MLT pass",
                          result.Error) ||
            !CreateBuffer(statistics,
                          size_t(std::max(MetalMlt::BootstrapBatch, chainCount)) * sizeof(GroupStatistics),
                          nullptr, "Arvo Metal MLT counters", result.Error) ||
            !CreateBuffer(failures, sizeof(uint32_t), nullptr, "Arvo Metal MLT numerical status",
                          result.Error))
            return false;
        MetalMlt::Controls controls{{emission.Center.X, emission.Center.Y, emission.Center.Z},
                                    emission.Radius,
                                    uint32_t(emission.Sources.size()),
                                    options.BootstrapSamples,
                                    chainCount,
                                    options.Seed ^ constants.SamplingSeed,
                                    0,
                                    0,
                                    options.LargeStepProbability,
                                    options.MutationScale};
        auto dispatch = [&](uint32_t stage, uint32_t threads, uint32_t counterRows = 0) {
            if (cancellation() || !ResetTraversalError(result.Error))
                return false;
            @autoreleasepool
            {
                id<MTLCommandBuffer> command = [CommandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (!command || !encoder)
                {
                    result.Error = "Metal MLT could not create its command buffer or encoder.";
                    return false;
                }
                [encoder setComputePipelineState:MetropolisPipelines[stage]];
                [encoder setBuffer:Nodes.Handle offset:0 atIndex:0];
                [encoder setBuffer:PrimitiveRefs.Handle offset:0 atIndex:1];
                [encoder setBuffer:Triangles.Handle offset:0 atIndex:2];
                [encoder setBuffer:Spheres.Handle offset:0 atIndex:3];
                [encoder setBuffer:Materials.Handle offset:0 atIndex:4];
                [encoder setBuffer:emitters.Handle offset:0 atIndex:5];
                [encoder setBytes:&constants length:sizeof(constants) atIndex:6];
                [encoder setBuffer:Accumulation.Handle offset:0 atIndex:7];
                [encoder setBuffer:SampleCounts.Handle offset:0 atIndex:8];
                [encoder setBuffer:statistics.Handle offset:0 atIndex:9];
                [encoder setBuffer:TextureDescriptors.Handle offset:0 atIndex:10];
                [encoder setBuffer:TextureTexels.Handle offset:0 atIndex:11];
                [encoder setBuffer:SampleTargets.Handle offset:0 atIndex:12];
                BindInstanceBuffers(encoder);
                [encoder setBuffer:chains.Handle offset:0 atIndex:MetalMlt::ChainBinding];
                [encoder setBuffer:pass.Handle offset:0 atIndex:MetalMlt::PassBinding];
                [encoder setBytes:&controls length:sizeof(controls) atIndex:MetalMlt::ControlsBinding];
                [encoder setBuffer:failures.Handle offset:0 atIndex:MetalMlt::ErrorBinding];
                [encoder setBuffer:bootstrap.Handle offset:0 atIndex:MetalMlt::BootstrapBinding];
                [encoder setBuffer:normalization.Handle offset:0 atIndex:MetalMlt::DistributionBinding];
                uint32_t groups = (threads + MetalMlt::ThreadsPerGroup - 1) / MetalMlt::ThreadsPerGroup;
                [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(MetalMlt::ThreadsPerGroup, 1, 1)];
                [encoder endEncoding];
                // Completion publishes shared chain/CDF/film storage to the next
                // command and retains frame-local resources even during timeout.
                if (!CommitAndWait(command, "Metal MLT dispatch", cancellation, result.Error) ||
                    !CheckTraversalError(result.Error))
                    return false;
            }
            const void* status = failures.Handle.contents;
            const auto* counters = static_cast<const GroupStatistics*>(statistics.Handle.contents);
            if (!status || (counterRows && !counters))
            {
                result.Error = "Metal MLT numerical status or counters are not CPU-visible.";
                return false;
            }
            uint32_t flags;
            std::memcpy(&flags, status, sizeof(flags));
            if (flags)
            {
                result.Error = "Metal MLT encountered invalid path, chain or film arithmetic.";
                return false;
            }
            for (uint32_t row = 0; row < counterRows; ++row)
            {
                result.Statistics.PrimaryRays += counters[row].PrimaryRays.Value();
                result.Statistics.ShadowRays += counters[row].ShadowRays.Value();
                result.Statistics.ReflectionRays += counters[row].ReflectionRays.Value();
                result.Statistics.PrimitiveTests += counters[row].PrimitiveTests.Value();
            }
            return true;
        };
        if (!dispatch(MetalMlt::ClearStage, pixels))
            return false;
        for (uint32_t first = 0; first < options.BootstrapSamples; first += MetalMlt::BootstrapBatch)
        {
            controls.First = first;
            controls.Count = std::min(MetalMlt::BootstrapBatch, options.BootstrapSamples - first);
            if (!dispatch(MetalMlt::BootstrapStage, controls.Count, controls.Count))
                return false;
        }
        if (!dispatch(MetalMlt::DistributionStage, 1) || !dispatch(MetalMlt::InitializeStage, chainCount))
            return false;
        // Crops replay the global chain schedule and keep only local splats.
        // A complete mutation pass, including untouched pixels, is one film sample.
        uint32_t steps = fullPixels / chainCount + (fullPixels % chainCount != 0);
        for (uint32_t sample = 0; sample < constants.TotalSamples; ++sample)
        {
            constants.SampleBegin = sample;
            if (sample && !dispatch(MetalMlt::ClearStage, pixels))
                return false;
            for (uint32_t first = 0; first < steps; first += MetalMlt::MutationBatch)
            {
                controls.First = first;
                controls.Count = std::min(MetalMlt::MutationBatch, steps - first);
                if (!dispatch(MetalMlt::MutateStage, chainCount, chainCount))
                    return false;
            }
            if (!dispatch(MetalMlt::CommitStage, pixels))
                return false;
        }
        return true;
    }

    bool EnsureProgressivePipelines(std::string& error) const
    {
        if (ProgressivePipelines[MetalPpm::ClearStage])
            return true;
        @autoreleasepool
        {
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            options.fastMathEnabled = NO;
            if (@available(macOS 11.0, *))
                options.languageVersion = MTLLanguageVersion2_3;
            else
            {
                ReleaseObject(options);
                error = "Metal PPM requires macOS 11 or later for its MSL 2.3 arithmetic leaves.";
                return false;
            }
            NSError* detail = nil;
            id<MTLLibrary> library =
                [Device newLibraryWithSource:[NSString stringWithUTF8String:Detail::MetalPpmSource]
                                     options:options
                                       error:&detail];
            ReleaseObject(options);
            if (!library)
            {
                error = ErrorDescription(detail, "Metal PPM compilation failed.");
                return false;
            }
            NSString* names[MetalPpm::StageCount] = {@"ArvoPpmClear", @"ArvoPpmInitialize", @"ArvoPpmCamera",
                                                     @"ArvoPpmEmit",  @"ArvoPpmIndex",      @"ArvoPpmGather",
                                                     @"ArvoPpmUpdate"};
            id<MTLComputePipelineState> compiled[MetalPpm::StageCount] = {};
            bool complete = true;
            for (uint32_t stage = 0; stage < MetalPpm::StageCount && complete; ++stage)
            {
                id<MTLFunction> function = [library newFunctionWithName:names[stage]];
                if (function)
                    compiled[stage] = [Device newComputePipelineStateWithFunction:function error:&detail];
                ReleaseObject(function);
                if (!compiled[stage])
                {
                    error = ErrorDescription(detail, "Metal PPM pipeline creation failed.");
                    complete = false;
                }
                else if (compiled[stage].maxTotalThreadsPerThreadgroup < MetalPpm::ThreadsPerGroup)
                {
                    error = "Metal PPM pipeline does not support its threadgroup size.";
                    complete = false;
                }
            }
            ReleaseObject(library);
            if (!complete)
            {
                for (auto pipeline : compiled)
                    ReleaseObject(pipeline);
                return false;
            }
            for (uint32_t stage = 0; stage < MetalPpm::StageCount; ++stage)
                ProgressivePipelines[stage] = compiled[stage];
            return true;
        }
    }
    bool RenderProgressivePhotons(const Scene& scene, FrameConstants constants,
                                  const ProgressivePhotonOptions& options,
                                  const Detail::CancellationProbe& cancellation, RenderResult& result) const
    {
        if (!EnsureProgressivePipelines(result.Error))
            return false;
        const auto emission = MetalEmission::PrepareEmitters(scene);
        for (const auto& source : emission.Sources)
            if (!(source.Flags & EmitterCastsShadows))
            {
                result.Error = "Metal PPM requires shadow casting on every emitter.";
                return false;
            }
        const uint32_t pixelCount = constants.Width * constants.Height;
        const uint32_t capacity = std::min(options.PathsPerPass, MetalPpm::LightBatch);
        const uint32_t slotsPerPath = constants.MaxReflectionBounces + 1;
        uint32_t buckets = 1;
        while (buckets < capacity)
            buckets *= 2;
        const size_t photonBytes = size_t(capacity) * slotsPerPath * MetalPpm::PhotonStride;
        const size_t visibleBytes = size_t(pixelCount) * MetalPpm::VisibleStride;
        const size_t pixelBytes = size_t(pixelCount) * MetalPpm::PixelStride;
        if (std::max({photonBytes, visibleBytes, pixelBytes}) > Device.maxBufferLength)
        {
            result.Error =
                "Metal PPM persistent or batch storage exceeds the device's maximum buffer length.";
            return false;
        }
        Buffer sources, photons, heads, visible, pixels, statistics, errors;
        constexpr uint32_t CounterRows = std::max(MetalPpm::LightBatch, MetalPpm::PixelBatch);
        if (!CreateVectorBuffer(sources, emission.Sources, "Arvo Metal PPM emitters", result.Error) ||
            !CreateBuffer(photons, photonBytes, nullptr, "Arvo Metal PPM photons", result.Error) ||
            !CreateBuffer(heads, size_t(buckets) * sizeof(uint32_t), nullptr, "Arvo Metal PPM grid",
                          result.Error) ||
            !CreateBuffer(visible, visibleBytes, nullptr, "Arvo Metal PPM visible points", result.Error) ||
            !CreateBuffer(pixels, pixelBytes, nullptr, "Arvo Metal PPM persistent state", result.Error) ||
            !CreateBuffer(statistics, size_t(CounterRows) * sizeof(GroupStatistics), nullptr,
                          "Arvo Metal PPM statistics", result.Error) ||
            !CreateBuffer(errors, sizeof(uint32_t), nullptr, "Arvo Metal PPM error", result.Error))
            return false;
        constexpr uint32_t CausticsEnabled = 1, IndirectEnabled = 2;
        MetalPpm::Controls controls{};
        controls.Center[0] = emission.Center.X;
        controls.Center[1] = emission.Center.Y;
        controls.Center[2] = emission.Center.Z;
        controls.WorldRadius = emission.Radius;
        controls.Sources = uint32_t(emission.Sources.size());
        controls.PathsPerPass = options.PathsPerPass;
        controls.SlotsPerPath = slotsPerPath;
        controls.Buckets = buckets;
        controls.Seed = options.Seed ^ constants.SamplingSeed;
        controls.Classes =
            (options.Caustics ? CausticsEnabled : 0) | (options.Indirect ? IndirectEnabled : 0);
        controls.InitialRadiusScale = options.InitialRadiusScale;
        controls.Alpha = options.Alpha;
        const auto dispatch = [&](uint32_t stage, uint32_t first, uint32_t count) {
            if (cancellation() || !ResetTraversalError(result.Error))
                return false;
            controls.First = first;
            controls.Count = count;
            @autoreleasepool
            {
                id<MTLCommandBuffer> command = [CommandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (!command || !encoder)
                {
                    result.Error = "Metal PPM command allocation failed.";
                    return false;
                }
                [encoder setComputePipelineState:ProgressivePipelines[stage]];
                [encoder setBuffer:Nodes.Handle offset:0 atIndex:0];
                [encoder setBuffer:PrimitiveRefs.Handle offset:0 atIndex:1];
                [encoder setBuffer:Triangles.Handle offset:0 atIndex:2];
                [encoder setBuffer:Spheres.Handle offset:0 atIndex:3];
                [encoder setBuffer:Materials.Handle offset:0 atIndex:4];
                [encoder setBuffer:sources.Handle offset:0 atIndex:5];
                [encoder setBytes:&constants length:sizeof(constants) atIndex:6];
                [encoder setBuffer:Accumulation.Handle offset:0 atIndex:7];
                [encoder setBuffer:SampleCounts.Handle offset:0 atIndex:8];
                [encoder setBuffer:statistics.Handle offset:0 atIndex:9];
                [encoder setBuffer:TextureDescriptors.Handle offset:0 atIndex:10];
                [encoder setBuffer:TextureTexels.Handle offset:0 atIndex:11];
                [encoder setBuffer:SampleTargets.Handle offset:0 atIndex:12];
                BindInstanceBuffers(encoder);
                [encoder setBuffer:photons.Handle offset:0 atIndex:MetalPpm::PhotonsBinding];
                [encoder setBuffer:heads.Handle offset:0 atIndex:MetalPpm::HeadsBinding];
                [encoder setBytes:&controls length:sizeof(controls) atIndex:MetalPpm::ControlsBinding];
                [encoder setBuffer:errors.Handle offset:0 atIndex:MetalPpm::ErrorBinding];
                [encoder setBuffer:visible.Handle offset:0 atIndex:MetalPpm::VisibleBinding];
                [encoder setBuffer:pixels.Handle offset:0 atIndex:MetalPpm::PixelsBinding];
                const uint32_t groups = (count + MetalPpm::ThreadsPerGroup - 1) / MetalPpm::ThreadsPerGroup;
                [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(MetalPpm::ThreadsPerGroup, 1, 1)];
                [encoder endEncoding];
                // Completed commands publish the hash links and wide state before
                // buffer reuse. Cancellation also waits for outstanding GPU work.
                if (!CommitAndWait(command, "Metal PPM dispatch", cancellation, result.Error) ||
                    !CheckTraversalError(result.Error))
                    return false;
            }
            const auto* failure = static_cast<const uint32_t*>(errors.Handle.contents);
            if (!failure || *failure)
            {
                result.Error = "Metal PPM kernel reported invalid configuration or numerical state.";
                return false;
            }
            if (stage == MetalPpm::CameraStage || stage == MetalPpm::EmitStage ||
                stage == MetalPpm::GatherStage)
            {
                const auto* counters = static_cast<const GroupStatistics*>(statistics.Handle.contents);
                if (!counters)
                {
                    result.Error = "Metal PPM statistics readback is unavailable.";
                    return false;
                }
                for (uint32_t row = 0; row < count; ++row)
                {
                    result.Statistics.PrimaryRays += counters[row].PrimaryRays.Value();
                    result.Statistics.ShadowRays += counters[row].ShadowRays.Value();
                    result.Statistics.ReflectionRays += counters[row].ReflectionRays.Value();
                    result.Statistics.PrimitiveTests += counters[row].PrimitiveTests.Value();
                }
            }
            return true;
        };
        const auto batch = [&](uint32_t stage, uint32_t total, uint32_t capacity) {
            for (uint32_t first = 0; first < total; first += capacity)
                if (!dispatch(stage, first, std::min(capacity, total - first)))
                    return false;
            return true;
        };
        if (!batch(MetalPpm::ClearStage, buckets, MetalPpm::LinearBatch) ||
            !batch(MetalPpm::InitializeStage, pixelCount, MetalPpm::LinearBatch))
            return false;
        for (controls.Pass = 0; controls.Pass < constants.TotalSamples; ++controls.Pass)
        {
            if (!batch(MetalPpm::CameraStage, pixelCount, MetalPpm::PixelBatch))
                return false;
            for (uint32_t first = 0; first < options.PathsPerPass; first += capacity)
            {
                controls.LightBegin = first;
                controls.LightCount = std::min(capacity, options.PathsPerPass - first);
                if (!batch(MetalPpm::ClearStage, buckets, MetalPpm::LinearBatch) ||
                    !dispatch(MetalPpm::EmitStage, 0, controls.LightCount) ||
                    !batch(MetalPpm::IndexStage, controls.LightCount * slotsPerPath, MetalPpm::LinearBatch) ||
                    !batch(MetalPpm::GatherStage, pixelCount, MetalPpm::PixelBatch))
                    return false;
            }
            if (!batch(MetalPpm::UpdateStage, pixelCount, MetalPpm::LinearBatch))
                return false;
        }
        return true;
    }

    bool EnsurePhotonPipelines(std::string& error) const
    {
        if (PhotonPipelines[MetalPhoton::ClearStage])
            return true;
        @autoreleasepool
        {
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            // WideMath uses rounded products and explicit FMA residuals.
            options.fastMathEnabled = NO;
            if (@available(macOS 11.0, *))
                options.languageVersion = MTLLanguageVersion2_3;
            else
            {
                ReleaseObject(options);
                error = "Metal photon mapping requires macOS 11 or later for its MSL 2.3 arithmetic leaves.";
                return false;
            }
            NSError* detail = nil;
            id<MTLLibrary> library =
                [Device newLibraryWithSource:[NSString stringWithUTF8String:Detail::MetalPhotonSource]
                                     options:options
                                       error:&detail];
            ReleaseObject(options);
            if (!library)
            {
                error = ErrorDescription(detail, "Metal photon compilation failed.");
                return false;
            }
            NSString* entries[MetalPhoton::StageCount] = {@"ArvoPhotonClear", @"ArvoPhotonEmit",
                                                          @"ArvoPhotonIndex", @"ArvoPhotonCamera"};
            id<MTLComputePipelineState> compiled[MetalPhoton::StageCount] = {};
            bool complete = true;
            for (uint32_t stage = 0; stage < MetalPhoton::StageCount && complete; ++stage)
            {
                id<MTLFunction> entry = [library newFunctionWithName:entries[stage]];
                if (entry)
                    compiled[stage] = [Device newComputePipelineStateWithFunction:entry error:&detail];
                ReleaseObject(entry);
                if (!compiled[stage])
                {
                    error = ErrorDescription(detail, "Metal photon pipeline creation failed.");
                    complete = false;
                }
                else if (compiled[stage].maxTotalThreadsPerThreadgroup < MetalPhoton::ThreadsPerGroup)
                {
                    error = "Metal photon pipeline does not support its threadgroup size.";
                    complete = false;
                }
            }
            ReleaseObject(library);
            if (!complete)
            {
                for (auto pipeline : compiled)
                    ReleaseObject(pipeline);
                return false;
            }
            for (uint32_t stage = 0; stage < MetalPhoton::StageCount; ++stage)
                PhotonPipelines[stage] = compiled[stage];
            return true;
        }
    }
    bool RenderPhotons(const Scene& scene, FrameConstants constants, const PhotonMappingOptions& options,
                       const Detail::CancellationProbe& cancellation, RenderResult& result) const
    {
        if (!EnsurePhotonPipelines(result.Error))
            return false;
        const auto emission = MetalEmission::PrepareEmitters(scene);
        for (const auto& source : emission.Sources)
            if (!(source.Flags & EmitterCastsShadows))
            {
                result.Error = "Metal photon mapping requires shadow casting on every emitter.";
                return false;
            }
        uint32_t buckets = 1;
        while (buckets < options.PhotonPaths)
            buckets *= 2;
        const uint32_t pixels = constants.Width * constants.Height,
                       slotsPerPath = constants.MaxReflectionBounces + 1;
        const uint32_t slotCount = options.PhotonPaths * slotsPerPath;
        const size_t recordBytes = size_t(slotCount) * sizeof(MetalPhoton::Photon);
        if (recordBytes > Device.maxBufferLength)
        {
            result.Error = "The requested Metal photon map exceeds this device's maximum buffer length.";
            return false;
        }
        Buffer emitters, records, heads, statistics, failure;
        if (!CreateVectorBuffer(emitters, emission.Sources, "Arvo Metal photon emitters", result.Error) ||
            !CreateBuffer(records, recordBytes, nullptr, "Arvo Metal photons", result.Error) ||
            !CreateBuffer(heads, size_t(buckets) * sizeof(uint32_t), nullptr, "Arvo Metal photon grid",
                          result.Error) ||
            !CreateBuffer(statistics, size_t(MetalPhoton::TransportBatch) * sizeof(GroupStatistics), nullptr,
                          "Arvo Metal photon counters", result.Error) ||
            !CreateBuffer(failure, sizeof(uint32_t), nullptr, "Arvo Metal photon status", result.Error))
            return false;
        constexpr uint32_t CausticsEnabled = 1, IndirectEnabled = 2;
        MetalPhoton::Controls controls{{emission.Center.X, emission.Center.Y, emission.Center.Z},
                                       emission.Radius,
                                       uint32_t(emission.Sources.size()),
                                       options.PhotonPaths,
                                       slotsPerPath,
                                       buckets,
                                       options.Seed ^ constants.SamplingSeed,
                                       (options.Caustics ? CausticsEnabled : 0) |
                                           (options.Indirect ? IndirectEnabled : 0),
                                       0,
                                       0,
                                       options.RadiusScale,
                                       0,
                                       {}};
        auto dispatch = [&](uint32_t stage, uint32_t threads, uint32_t counterRows = 0) {
            if (cancellation() || !ResetTraversalError(result.Error))
                return false;
            @autoreleasepool
            {
                id<MTLCommandBuffer> command = [CommandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (!command || !encoder)
                {
                    result.Error = "Metal photon command allocation failed.";
                    return false;
                }
                [encoder setComputePipelineState:PhotonPipelines[stage]];
                [encoder setBuffer:Nodes.Handle offset:0 atIndex:0];
                [encoder setBuffer:PrimitiveRefs.Handle offset:0 atIndex:1];
                [encoder setBuffer:Triangles.Handle offset:0 atIndex:2];
                [encoder setBuffer:Spheres.Handle offset:0 atIndex:3];
                [encoder setBuffer:Materials.Handle offset:0 atIndex:4];
                [encoder setBuffer:emitters.Handle offset:0 atIndex:5];
                [encoder setBytes:&constants length:sizeof(constants) atIndex:6];
                [encoder setBuffer:Accumulation.Handle offset:0 atIndex:7];
                [encoder setBuffer:SampleCounts.Handle offset:0 atIndex:8];
                [encoder setBuffer:statistics.Handle offset:0 atIndex:9];
                [encoder setBuffer:TextureDescriptors.Handle offset:0 atIndex:10];
                [encoder setBuffer:TextureTexels.Handle offset:0 atIndex:11];
                [encoder setBuffer:SampleTargets.Handle offset:0 atIndex:12];
                BindInstanceBuffers(encoder);
                [encoder setBuffer:records.Handle offset:0 atIndex:MetalPhoton::RecordsBinding];
                [encoder setBuffer:heads.Handle offset:0 atIndex:MetalPhoton::HeadsBinding];
                [encoder setBytes:&controls length:sizeof(controls) atIndex:MetalPhoton::ControlsBinding];
                [encoder setBuffer:failure.Handle offset:0 atIndex:MetalPhoton::ErrorBinding];
                uint32_t groups = (threads + MetalPhoton::ThreadsPerGroup - 1) / MetalPhoton::ThreadsPerGroup;
                [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(MetalPhoton::ThreadsPerGroup, 1, 1)];
                [encoder endEncoding];
                // Complete each bounded pass before reading shared status, reusing
                // controls, querying the immutable map or releasing local buffers.
                if (!CommitAndWait(command, "Metal photon dispatch", cancellation, result.Error) ||
                    !CheckTraversalError(result.Error))
                    return false;
            }
            const auto* status = static_cast<const uint32_t*>(failure.Handle.contents);
            const auto* counters = static_cast<const GroupStatistics*>(statistics.Handle.contents);
            if (!status || (counterRows && !counters))
            {
                result.Error = "Metal photon readback is unavailable.";
                return false;
            }
            if (*status)
            {
                result.Error = "Metal photon kernel reported invalid configuration or numerical state.";
                return false;
            }
            for (uint32_t row = 0; row < counterRows; ++row)
            {
                result.Statistics.PrimaryRays += counters[row].PrimaryRays.Value();
                result.Statistics.ShadowRays += counters[row].ShadowRays.Value();
                result.Statistics.ReflectionRays += counters[row].ReflectionRays.Value();
                result.Statistics.PrimitiveTests += counters[row].PrimitiveTests.Value();
            }
            return true;
        };
        if (!dispatch(MetalPhoton::ClearStage, buckets))
            return false;
        for (uint32_t first = 0; first < options.PhotonPaths; first += MetalPhoton::TransportBatch)
        {
            controls.First = first;
            controls.Count = std::min(MetalPhoton::TransportBatch, options.PhotonPaths - first);
            if (!dispatch(MetalPhoton::EmitStage, controls.Count, controls.Count))
                return false;
        }
        for (uint32_t first = 0; first < slotCount; first += MetalPhoton::IndexBatch)
        {
            controls.First = first;
            controls.Count = std::min(MetalPhoton::IndexBatch, slotCount - first);
            if (!dispatch(MetalPhoton::IndexStage, controls.Count))
                return false;
        }
        // Camera tiles share the full-scene emission seed/budget and disk radius.
        for (controls.Sample = 0; controls.Sample < constants.TotalSamples; ++controls.Sample)
            for (uint32_t first = 0; first < pixels; first += MetalPhoton::TransportBatch)
            {
                controls.First = first;
                controls.Count = std::min(MetalPhoton::TransportBatch, pixels - first);
                if (!dispatch(MetalPhoton::CameraStage, controls.Count, controls.Count))
                    return false;
            }
        return true;
    }

    bool EnsureVcmPipelines(std::string& error) const
    {
        if (VcmPipelines[0] != nil)
            return true;
        if (@available(macOS 11.0, *))
        {
            NSError* detail = nil;
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            options.languageVersion = MTLLanguageVersion2_3;
            options.fastMathEnabled = NO;
            id<MTLLibrary> library =
                [Device newLibraryWithSource:[NSString stringWithUTF8String:Detail::MetalVcmSource]
                                     options:options
                                       error:&detail];
            ReleaseObject(options);
            if (library == nil)
            {
                error = ErrorDescription(detail, "Metal VCM compilation failed.");
                return false;
            }
            id<MTLComputePipelineState> compiled[MetalVcm::StageCount] = {};
            bool valid = true;
            for (uint32_t stage = 0; stage < MetalVcm::StageCount && valid; ++stage)
            {
                id<MTLFunction> function = [library
                    newFunctionWithName:[NSString stringWithUTF8String:MetalVcm::EntryPoints[stage]]];
                if (function != nil)
                    compiled[stage] = [Device newComputePipelineStateWithFunction:function error:&detail];
                ReleaseObject(function);
                if (compiled[stage] == nil ||
                    compiled[stage].maxTotalThreadsPerThreadgroup < MetalVcm::ThreadsPerGroup)
                {
                    valid = false;
                    error = std::string(MetalVcm::EntryPoints[stage]) + ": " +
                            ErrorDescription(detail, "Metal VCM pipeline cannot support its threadgroup.");
                }
            }
            ReleaseObject(library);
            for (uint32_t stage = 0; stage < MetalVcm::StageCount; ++stage)
                if (valid)
                    VcmPipelines[stage] = compiled[stage];
                else
                    ReleaseObject(compiled[stage]);
            return valid;
        }
        error = "Metal VCM requires macOS 11 or later for MSL 2.3 arithmetic.";
        return false;
    }

    bool RenderVcm(const Scene& scene, const FrameConstants& camera, const RenderSettings& settings,
                   const Detail::CancellationProbe& cancellation, RenderResult& result) const
    {
        using namespace MetalVcm;
        if (cancellation())
            return false;
        const auto emission = MetalEmission::PrepareEmitters(scene);
        if (emission.Sources.size() > std::numeric_limits<uint32_t>::max() ||
            emission.Sources.size() > Device.maxBufferLength / sizeof(PackedEmitter))
        {
            result.Error = "Metal VCM source storage exceeds the device limit.";
            return false;
        }
        for (const auto& source : emission.Sources)
            if (!(source.Flags & EmitterCastsShadows))
            {
                result.Error = "Metal VCM requires shadow casting on every emitter.";
                return false;
            }
        FrameRequest request;
        auto& c = request.Scene;
        auto copyVector = [](float(&out)[3], Vec3 value) {
            out[0] = value.X;
            out[1] = value.Y;
            out[2] = value.Z;
        };
        copyVector(c.CameraPosition, camera.CameraPosition);
        copyVector(c.CameraForward, camera.CameraForward);
        copyVector(c.CameraRight, camera.CameraRight);
        copyVector(c.CameraUp, camera.CameraUp);
        copyVector(c.EmissionCenter, ToVec3(emission.Center));
        c.FullWidth = camera.FullWidth;
        c.FullHeight = camera.FullHeight;
        c.TotalSamples = camera.TotalSamples;
        c.MaxReflections = camera.MaxReflectionBounces;
        c.HalfWidth = camera.HalfWidth;
        c.HalfHeight = camera.HalfHeight;
        c.LensRadius = camera.LensRadius;
        c.FocusDistance = camera.FocusDistance;
        c.MaterialCount = MaterialCount;
        c.NodeCount = NodeCount;
        c.TriangleCount = TriangleCount;
        c.Acceleration = camera.Acceleration;
        c.InstanceNodeCount = InstanceNodeCount;
        c.TextureCount = TextureCount;
        c.TexelCount = TextureTexelCount;
        c.TransportFlags = camera.TransportFlags;
        c.EmissionRadius = emission.Radius;
        c.SourceCount = uint32_t(emission.Sources.size());
        c.Sequence = camera.SamplingSequence;
        c.SamplingSeed = camera.SamplingSeed;
        request.TileX = camera.TileX;
        request.TileY = camera.TileY;
        request.Width = camera.Width;
        request.Height = camera.Height;
        request.Options = settings.Vcm;
        request.Mis = settings.Transport.Mis;
        request.NextEventEstimation = settings.Transport.NextEventEstimation;
        request.MaximumBufferBytes = Device.maxBufferLength;
        FramePlan plan;
        if (!PrepareFrame(request, plan, result.Error) || !EnsureVcmPipelines(result.Error) || cancellation())
            return false;
        std::array<Buffer, StorageCount> storage;
        Buffer sources;
        if (!CreateVectorBuffer(sources, emission.Sources, "Arvo Metal VCM sources", result.Error))
            return false;
        const char* labels[StorageCount] = {
            "Arvo Metal VCM vertices", "Arvo Metal VCM paths",       "Arvo Metal VCM statistics",
            "Arvo Metal VCM index",    "Arvo Metal VCM pixel state", "Arvo Metal VCM splats",
            "Arvo Metal VCM film",     "Arvo Metal VCM pending",     "Arvo Metal VCM errors"};
        for (uint32_t i = 0; i < StorageCount; ++i)
        {
            if (cancellation() || !CreateBuffer(storage[i], plan.Bytes[i], nullptr, labels[i], result.Error))
                return false;
            if (!storage[i].Handle.contents)
            {
                result.Error = "Metal VCM storage is not CPU-visible.";
                return false;
            }
        }
        auto buffer = [&](Storage slot) { return storage[uint32_t(slot)].Handle; };
        auto* failure = static_cast<uint32_t*>(buffer(Storage::Errors).contents);
        auto* pending = static_cast<uint32_t*>(buffer(Storage::Pending).contents);
        *failure = *pending = 0;
        FrameCallbacks callbacks;
        callbacks.Cancelled = cancellation;
        callbacks.Execute = [&](const MetalVcm::Dispatch& dispatch, uint32_t& continuation,
                                std::string& error) {
            @autoreleasepool
            {
                *pending = 0;
                id<MTLCommandBuffer> command = [CommandQueue commandBuffer];
                // Default command buffers retain bound resources through completion,
                // including a terminal timeout. Never use unretained references here.
                // https://developer.apple.com/documentation/metal/mtlcommandbuffer/retainedreferences
                if (command == nil || !command.retainedReferences)
                {
                    error = "Metal VCM requires a retaining command buffer.";
                    return false;
                }
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (encoder == nil)
                {
                    error = "Metal VCM command encoder allocation failed.";
                    return false;
                }
                [encoder setComputePipelineState:VcmPipelines[uint32_t(dispatch.Kernel)]];
                [encoder setBuffer:Nodes.Handle offset:0 atIndex:NodesBinding];
                [encoder setBuffer:PrimitiveRefs.Handle offset:0 atIndex:ReferencesBinding];
                [encoder setBuffer:Triangles.Handle offset:0 atIndex:TrianglesBinding];
                [encoder setBuffer:Spheres.Handle offset:0 atIndex:SpheresBinding];
                [encoder setBuffer:Materials.Handle offset:0 atIndex:MaterialsBinding];
                [encoder setBuffer:sources.Handle offset:0 atIndex:SourcesBinding];
                [encoder setBytes:&plan.Scene length:sizeof(plan.Scene) atIndex:SceneBinding];
                [encoder setBuffer:TextureDescriptors.Handle offset:0 atIndex:TexturesBinding];
                [encoder setBuffer:TextureTexels.Handle offset:0 atIndex:TexelsBinding];
                [encoder setBuffer:InstanceRecords.Handle offset:0 atIndex:InstancesBinding];
                [encoder setBuffer:InstanceNodes.Handle offset:0 atIndex:MetalVcm::InstanceNodesBinding];
                [encoder setBuffer:buffer(Storage::Errors) offset:0 atIndex:ErrorsBinding];
                if (dispatch.Kernel == Stage::LightWalks || dispatch.Kernel == Stage::CameraWalks)
                    [encoder setBytes:&dispatch.Walks length:sizeof(dispatch.Walks) atIndex:ControlsBinding];
                else
                    [encoder setBytes:&dispatch.Coupling
                               length:sizeof(dispatch.Coupling)
                              atIndex:ControlsBinding];
                [encoder setBuffer:buffer(Storage::Vertices) offset:0 atIndex:VerticesBinding];
                [encoder setBuffer:buffer(Storage::Paths) offset:0 atIndex:PathsBinding];
                [encoder setBuffer:buffer(Storage::Statistics) offset:0 atIndex:StatisticsBinding];
                [encoder setBytes:&dispatch.Frame length:sizeof(dispatch.Frame) atIndex:FrameBinding];
                [encoder setBuffer:buffer(Storage::Index) offset:0 atIndex:IndexBinding];
                [encoder setBuffer:buffer(Storage::Pixels) offset:0 atIndex:PixelsBinding];
                [encoder setBuffer:buffer(Storage::Splats) offset:0 atIndex:SplatsBinding];
                [encoder setBuffer:buffer(Storage::Film) offset:0 atIndex:FilmBinding];
                [encoder setBuffer:buffer(Storage::Pending) offset:0 atIndex:PendingBinding];
                uint32_t groups =
                    (dispatch.Lanes + MetalVcm::ThreadsPerGroup - 1) / MetalVcm::ThreadsPerGroup;
                [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(MetalVcm::ThreadsPerGroup, 1, 1)];
                [encoder endEncoding];
                if (!CommitAndWait(command, MetalVcm::EntryPoints[uint32_t(dispatch.Kernel)], cancellation,
                                   error))
                    return false;
                if (*failure)
                {
                    error = "Metal VCM kernel error bits: " + std::to_string(*failure);
                    return false;
                }
                continuation = *pending;
                const auto* counters =
                    static_cast<const WalkStatistics*>(buffer(Storage::Statistics).contents);
                auto counter = [](const uint32_t(&parts)[2]) {
                    return uint64_t(parts[0]) | (uint64_t(parts[1]) << 32);
                };
                for (uint32_t lane = 0; lane < dispatch.StatisticsRows; ++lane)
                {
                    result.Statistics.PrimaryRays += counter(counters[lane].Camera);
                    result.Statistics.ReflectionRays += counter(counters[lane].Transport);
                    result.Statistics.PrimitiveTests += counter(counters[lane].Tests);
                    result.Statistics.ShadowRays += counter(counters[lane].Shadow);
                }
                return true;
            }
        };
        const auto* film = static_cast<const MetalVcm::FilmPixel*>(buffer(Storage::Film).contents);
        callbacks.Publish = [&](uint32_t completed, std::string& error) {
            for (uint32_t pixel = 0; pixel < plan.Frame.PixelCapacity; ++pixel)
            {
                if (pixel % LinearBatch == 0 && cancellation())
                    return false;
                if (film[pixel].Samples != completed)
                {
                    error = "Metal VCM did not commit a complete pass.";
                    return false;
                }
            }
            return true;
        };
        if (!ExecuteFrame(plan, callbacks, result.Error))
            return false;
        result.Film = LinearFilm(settings.Width, settings.Height);
        result.Output = Image(settings.Width, settings.Height);
        auto& rgba = result.Output.GetPixelsRGBA8();
        constexpr size_t Channels = 4;
        constexpr uint8_t Opaque = 255;
        for (uint32_t pixel = 0; pixel < plan.Frame.PixelCapacity; ++pixel)
        {
            if (pixel % LinearBatch == 0 && cancellation())
                return false;
            Float3 mean(film[pixel].Mean[0], film[pixel].Mean[1], film[pixel].Mean[2]);
            if (!IsFinite(mean) || !IsNonNegative(mean))
            {
                result.Error = "Metal VCM final film is not finite and nonnegative.";
                return false;
            }
            result.Film.Pixels[pixel] = {mean, 0, film[pixel].Samples};
            for (size_t channel = 0; channel < Channels - 1; ++channel)
                rgba[size_t(pixel) * Channels + channel] =
                    settings.GammaCorrection ? Maths::LinearToSRGB8(film[pixel].Mean[channel])
                                             : Maths::LinearToUNorm8(film[pixel].Mean[channel]);
            rgba[size_t(pixel) * Channels + Channels - 1] = Opaque;
            ++result.Statistics.CompletedPixels;
        }
        return !cancellation();
    }

    bool MicropolygonPassNative(MicropolygonData::Controls controls,
                                const std::vector<MicropolygonData::Patch>& patches,
                                const std::vector<MicropolygonData::Range>& ranges,
                                std::vector<uint32_t>& words, const Detail::CancellationProbe& cancelled,
                                std::string& error) const
    {
        namespace M = MicropolygonData;
        constexpr uint32_t Threads = 64;
        if (DicingPipeline == nil)
        {
            if (@available(macOS 11.0, *))
            {
                NSError* detail = nil;
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                options.languageVersion = MTLLanguageVersion2_3;
                options.fastMathEnabled = NO;
                id<MTLLibrary> library = [Device
                    newLibraryWithSource:[NSString stringWithUTF8String:Detail::MetalMicropolygonSource]
                                 options:options
                                   error:&detail];
                ReleaseObject(options);
                if (library == nil)
                {
                    error = ErrorDescription(detail, "Metal dicing compilation failed");
                    return false;
                }
                id<MTLFunction> function = [library newFunctionWithName:@"ArvoDiceMicropolygons"];
                id<MTLComputePipelineState> pipeline =
                    function == nil ? nil
                                    : [Device newComputePipelineStateWithFunction:function error:&detail];
                ReleaseObject(function);
                ReleaseObject(library);
                if (pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup < Threads)
                {
                    ReleaseObject(pipeline);
                    error = ErrorDescription(detail, "Metal dicing pipeline is unavailable");
                    return false;
                }
                DicingPipeline = pipeline;
            }
            else
            {
                error = "Metal micropolygons require macOS 11 or later";
                return false;
            }
        }
        const M::Range empty{};
        const void* sources[]{patches.data(), ranges.empty() ? &empty : ranges.data(), nullptr};
        const size_t sizes[]{patches.size() * sizeof(M::Patch),
                             std::max(size_t(1), ranges.size()) * sizeof(M::Range),
                             words.size() * sizeof(uint32_t)};
        constexpr uint32_t BufferCount = 3, Output = 2;
        Buffer buffers[BufferCount];
        for (uint32_t i = 0; i < BufferCount; ++i)
        {
            if (sizes[i] > Device.maxBufferLength)
            {
                error = "Metal micropolygon buffer exceeds device limits";
                return false;
            }
            if (cancelled() || !CreateBuffer(buffers[i], sizes[i], sources[i], "Arvo micropolygons", error))
                return false;
        }
        if (!buffers[Output].Handle.contents)
        {
            error = "Metal dicing output is not CPU-visible";
            return false;
        }
        std::memset(buffers[Output].Handle.contents, 0, sizes[Output]);
        constexpr uint32_t BatchItems = 16384;
        const uint32_t population = controls.Dispatch.Y;
        for (controls.Dispatch.X = 0; controls.Dispatch.X < population; controls.Dispatch.X += BatchItems)
        {
            if (cancelled())
                return false;
            controls.Dispatch.Y = std::min(BatchItems, population - controls.Dispatch.X);
            @autoreleasepool
            {
                id<MTLCommandBuffer> command = [CommandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (command == nil || encoder == nil)
                {
                    error = "Metal dicing command allocation failed";
                    return false;
                }
                [encoder setComputePipelineState:DicingPipeline];
                [encoder setBytes:&controls length:sizeof(controls) atIndex:0];
                for (NSUInteger i = 0; i < BufferCount; ++i)
                    [encoder setBuffer:buffers[i].Handle offset:0 atIndex:i + 1];
                [encoder dispatchThreadgroups:MTLSizeMake((controls.Dispatch.Y + Threads - 1) / Threads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(Threads, 1, 1)];
                [encoder endEncoding];
                if (!CommitAndWait(command, "Metal micropolygon generation", cancelled, error))
                    return false;
            }
        }
        std::memcpy(words.data(), buffers[Output].Handle.contents, sizes[Output]);
        return !cancelled();
    }

    bool EnsureTemporalPipeline(std::string& error) const
    {
        if (TemporalDenoisePipeline != nil)
            return true;
        if (@available(macOS 11.0, *))
        {
            NSError* detail = nil;
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            options.languageVersion = MTLLanguageVersion2_3;
            options.fastMathEnabled = NO;
            id<MTLLibrary> library = [Device
                newLibraryWithSource:[NSString stringWithUTF8String:Detail::MetalTemporalDenoiseSource]
                             options:options
                               error:&detail];
            ReleaseObject(options);
            if (library == nil)
            {
                error = ErrorDescription(detail, "Metal temporal compilation failed");
                return false;
            }
            id<MTLFunction> function = [library newFunctionWithName:@"ArvoTemporalDenoise"];
            id<MTLComputePipelineState> pipeline =
                function == nil ? nil : [Device newComputePipelineStateWithFunction:function error:&detail];
            ReleaseObject(function);
            ReleaseObject(library);
            constexpr uint32_t Threads = 64;
            if (pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup < Threads)
            {
                ReleaseObject(pipeline);
                error = ErrorDescription(detail, "Metal temporal pipeline is unavailable");
                return false;
            }
            TemporalDenoisePipeline = pipeline;
            return true;
        }
        error = "Metal temporal reconstruction requires macOS 11 or later";
        return false;
    }
    bool ReconstructTemporalNative(TemporalData::Controls controls, const std::vector<TemporalPixel>& current,
                                   const std::vector<GeometryGuide>& guides,
                                   const std::vector<TemporalPixel>& previous,
                                   const std::vector<GeometryGuide>& oldGuides,
                                   const Detail::CancellationProbe& cancelled,
                                   std::vector<TemporalPixel>& result, Image& display,
                                   std::string& error) const
    {
        if (!EnsureTemporalPipeline(error))
            return false;
        const TemporalPixel emptyColor{};
        const GeometryGuide emptyGuide{};
        constexpr uint32_t InputCount = 4, OutputCount = 3, ColorOutput = 0, DisplayOutput = 1,
                           ErrorOutput = 2;
        constexpr NSUInteger OutputBinding = InputCount, ControlsBinding = InputCount + OutputCount;
        const size_t count = size_t(controls.Width) * controls.Height,
                     colorBytes = count * sizeof(TemporalPixel), displayBytes = count * sizeof(uint32_t);
        const void* sources[] = {current.data(), guides.data(),
                                 previous.empty() ? &emptyColor : previous.data(),
                                 oldGuides.empty() ? &emptyGuide : oldGuides.data()};
        const size_t inputBytes[] = {
            current.size() * sizeof(current[0]), guides.size() * sizeof(guides[0]),
            previous.empty() ? sizeof(emptyColor) : previous.size() * sizeof(previous[0]),
            oldGuides.empty() ? sizeof(emptyGuide) : oldGuides.size() * sizeof(oldGuides[0])};
        const size_t outputBytes[] = {colorBytes, displayBytes, sizeof(uint32_t)};
        for (auto bytes : inputBytes)
            if (bytes > Device.maxBufferLength)
            {
                error = "Metal temporal input exceeds buffer limit";
                return false;
            }
        for (auto bytes : outputBytes)
            if (bytes > Device.maxBufferLength)
            {
                error = "Metal temporal output exceeds buffer limit";
                return false;
            }
        Buffer inputs[InputCount], outputs[OutputCount];
        for (uint32_t i = 0; i < InputCount; ++i)
            if (!CreateBuffer(inputs[i], inputBytes[i], sources[i], "Arvo temporal input", error))
                return false;
        for (uint32_t i = 0; i < OutputCount; ++i)
        {
            if (!CreateBuffer(outputs[i], outputBytes[i], nullptr, "Arvo temporal output", error))
                return false;
            if (!outputs[i].Handle.contents)
            {
                error = "Metal temporal output is not CPU-visible";
                return false;
            }
        }
        std::memset(outputs[ErrorOutput].Handle.contents, 0, sizeof(uint32_t));
        constexpr uint32_t Batch = 16384, Threads = 64;
        for (controls.Begin = 0; controls.Begin < count; controls.Begin += Batch)
        {
            if (cancelled())
                return false;
            controls.Count = std::min(Batch, uint32_t(count) - controls.Begin);
            @autoreleasepool
            {
                id<MTLCommandBuffer> command = [CommandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (command == nil || encoder == nil)
                {
                    error = "Metal temporal command allocation failed";
                    return false;
                }
                [encoder setComputePipelineState:TemporalDenoisePipeline];
                for (NSUInteger i = 0; i < InputCount; ++i)
                    [encoder setBuffer:inputs[i].Handle offset:0 atIndex:i];
                for (NSUInteger i = 0; i < OutputCount; ++i)
                    [encoder setBuffer:outputs[i].Handle offset:0 atIndex:OutputBinding + i];
                [encoder setBytes:&controls length:sizeof(controls) atIndex:ControlsBinding];
                [encoder dispatchThreadgroups:MTLSizeMake((controls.Count + Threads - 1) / Threads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(Threads, 1, 1)];
                [encoder endEncoding];
                if (!CommitAndWait(command, "Metal temporal reconstruction", cancelled, error))
                    return false;
            }
            uint32_t failure = 0;
            std::memcpy(&failure, outputs[ErrorOutput].Handle.contents, sizeof(failure));
            if (failure)
            {
                error = "Metal temporal reconstruction reported invalid configuration or radiance";
                return false;
            }
        }
        std::vector<TemporalPixel> filtered(count);
        Image encoded(controls.Width, controls.Height);
        std::memcpy(filtered.data(), outputs[ColorOutput].Handle.contents, colorBytes);
        std::memcpy(encoded.GetPixelsRGBA8().data(), outputs[DisplayOutput].Handle.contents, displayBytes);
        if (cancelled())
            return false;
        result = std::move(filtered);
        display = std::move(encoded);
        return true;
    }

    bool EnsureGeometryGuidePipelines(std::string& error) const
    {
        if (GeometryGuidePipelines[0] != nil && GeometryGuidePipelines[1] != nil)
            return true;
        if (@available(macOS 11.0, *))
        {
            NSError* detail = nil;
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            options.languageVersion = MTLLanguageVersion2_3;
            options.fastMathEnabled = NO;
            id<MTLLibrary> library =
                [Device newLibraryWithSource:[NSString stringWithUTF8String:Detail::MetalGeometryGuidesSource]
                                     options:options
                                       error:&detail];
            ReleaseObject(options);
            if (library == nil)
            {
                error = ErrorDescription(detail, "Metal geometry guide compilation failed");
                return false;
            }
            const std::array<NSString*, 2> names{@"ArvoGenerateGeometryGuideRays",
                                                 @"ArvoResolveGeometryGuideHits"};
            id<MTLComputePipelineState> compiled[2]{};
            constexpr uint32_t Threads = 64;
            bool success = true;
            for (size_t stage = 0; stage < names.size(); ++stage)
            {
                id<MTLFunction> function = [library newFunctionWithName:names[stage]];
                if (function != nil)
                    compiled[stage] = [Device newComputePipelineStateWithFunction:function error:&detail];
                ReleaseObject(function);
                if (compiled[stage] == nil || compiled[stage].maxTotalThreadsPerThreadgroup < Threads)
                {
                    success = false;
                    break;
                }
            }
            ReleaseObject(library);
            if (!success)
            {
                for (auto pipeline : compiled)
                    ReleaseObject(pipeline);
                error = ErrorDescription(detail, "Metal geometry guide pipeline creation failed");
                return false;
            }
            for (size_t stage = 0; stage < names.size(); ++stage)
                GeometryGuidePipelines[stage] = compiled[stage];
            return true;
        }
        error = "Metal geometry guides require macOS 11 or later";
        return false;
    }

    template <class Output>
    bool GeometryGuideStage(GeometryGuideData::Controls controls,
                            const std::vector<GeometryGuideData::EncodedRay>& rays,
                            const std::vector<RayQueryHit>& hits, const Detail::CancellationProbe& cancelled,
                            std::vector<Output>& result, std::string& error) const
    {
        constexpr bool Generate = std::is_same<Output, GeometryGuideData::EncodedRay>::value;
        if (!EnsureGeometryGuidePipelines(error))
            return false;
        const size_t count = size_t(controls.Width) * controls.Height, bytes = count * sizeof(Output);
        if (bytes > Device.maxBufferLength || rays.size() * sizeof(rays[0]) > Device.maxBufferLength ||
            hits.size() * sizeof(hits[0]) > Device.maxBufferLength)
        {
            error = "Metal geometry guides exceed the device buffer limit";
            return false;
        }
        Buffer rayInput, hitInput, output, errors;
        if constexpr (!Generate)
        {
            if (rays.size() != count || hits.size() != count)
            {
                error = "Metal geometry guide input extent mismatch";
                return false;
            }
            if (!CreateVectorBuffer(rayInput, rays, "Arvo geometry guide rays", error) ||
                !CreateVectorBuffer(hitInput, hits, "Arvo geometry guide hits", error))
                return false;
        }
        if (!CreateBuffer(output, bytes, nullptr, "Arvo geometry guide output", error) ||
            !CreateBuffer(errors, sizeof(uint32_t), nullptr, "Arvo geometry guide errors", error))
            return false;
        if (!errors.Handle.contents || !output.Handle.contents)
        {
            error = "Metal geometry guide buffers are not CPU-visible";
            return false;
        }
        std::memset(errors.Handle.contents, 0, sizeof(uint32_t));
        constexpr uint32_t Batch = 16384, Threads = 64;
        for (controls.Begin = 0; controls.Begin < count; controls.Begin += Batch)
        {
            if (cancelled())
                return false;
            controls.Count = std::min(Batch, uint32_t(count) - controls.Begin);
            @autoreleasepool
            {
                id<MTLCommandBuffer> command = [CommandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (command == nil || encoder == nil)
                {
                    error = "Metal geometry guide command allocation failed";
                    return false;
                }
                [encoder setComputePipelineState:GeometryGuidePipelines[Generate ? 0 : 1]];
                if constexpr (Generate)
                {
                    [encoder setBuffer:output.Handle offset:0 atIndex:0];
                    [encoder setBuffer:errors.Handle offset:0 atIndex:1];
                    [encoder setBytes:&controls length:sizeof(controls) atIndex:2];
                }
                else
                {
                    [encoder setBuffer:rayInput.Handle offset:0 atIndex:0];
                    [encoder setBuffer:hitInput.Handle offset:0 atIndex:1];
                    [encoder setBuffer:output.Handle offset:0 atIndex:2];
                    [encoder setBuffer:errors.Handle offset:0 atIndex:3];
                    [encoder setBytes:&controls length:sizeof(controls) atIndex:4];
                }
                [encoder dispatchThreadgroups:MTLSizeMake((controls.Count + Threads - 1) / Threads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(Threads, 1, 1)];
                [encoder endEncoding];
                if (!CommitAndWait(command, "Metal geometry guides", cancelled, error))
                    return false;
            }
            uint32_t failure = 0;
            std::memcpy(&failure, errors.Handle.contents, sizeof(failure));
            if (failure)
            {
                error = "Metal geometry guide generation or resolve failed";
                return false;
            }
        }
        if (cancelled())
            return false;
        std::vector<Output> decoded(count);
        std::memcpy(decoded.data(), output.Handle.contents, bytes);
        result = std::move(decoded);
        return true;
    }

    bool EnsureSpatialDenoisePipeline(std::string& error) const
    {
        if (SpatialDenoisePipeline != nil)
            return true;
        if (@available(macOS 11.0, *))
        {
            NSError* detail = nil;
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            options.languageVersion = MTLLanguageVersion2_3;
            options.fastMathEnabled = NO;
            id<MTLLibrary> library =
                [Device newLibraryWithSource:[NSString stringWithUTF8String:Detail::MetalSpatialDenoiseSource]
                                     options:options
                                       error:&detail];
            ReleaseObject(options);
            if (library == nil)
            {
                error = ErrorDescription(detail, "Metal spatial denoising compilation failed.");
                return false;
            }
            id<MTLFunction> function = [library newFunctionWithName:@"ArvoSpatialDenoise"];
            id<MTLComputePipelineState> pipeline =
                function == nil ? nil : [Device newComputePipelineStateWithFunction:function error:&detail];
            ReleaseObject(function);
            ReleaseObject(library);
            constexpr uint32_t Threads = 64;
            if (pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup < Threads)
            {
                error = ErrorDescription(detail, "Metal spatial denoising pipeline is unavailable.");
                ReleaseObject(pipeline);
                return false;
            }
            SpatialDenoisePipeline = pipeline;
            return true;
        }
        error = "Metal spatial denoising requires macOS 11 or later.";
        return false;
    }

    bool AllocateDenoisedNative(const std::vector<AdaptiveReconstructionData::Pixel>& input,
        AdaptiveReconstructionData::Controls controls, const Detail::CancellationProbe& cancellation,
        std::vector<uint32_t>& result, std::string& error) const
    {
        constexpr uint32_t Threads = 64, Batch = 16384;
        if (AdaptiveDenoisePipeline == nil)
        {
            if (@available(macOS 11.0, *))
            {
                NSError* detail = nil;
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                options.languageVersion = MTLLanguageVersion2_3;
                options.fastMathEnabled = NO;
                id<MTLLibrary> library = [Device newLibraryWithSource:
                    [NSString stringWithUTF8String:Detail::MetalAdaptiveDenoiseSource] options:options error:&detail];
                ReleaseObject(options);
                if (library == nil) { error = ErrorDescription(detail, "Metal adaptive allocation compilation failed."); return false; }
                id<MTLFunction> function = [library newFunctionWithName:@"ArvoAdaptiveDenoise"];
                id<MTLComputePipelineState> pipeline = function == nil ? nil :
                    [Device newComputePipelineStateWithFunction:function error:&detail];
                ReleaseObject(function); ReleaseObject(library);
                if (pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup < Threads)
                {
                    error = ErrorDescription(detail, "Metal adaptive allocation pipeline is unavailable.");
                    ReleaseObject(pipeline); return false;
                }
                AdaptiveDenoisePipeline = pipeline;
            }
            else { error = "Metal adaptive allocation requires macOS 11 or later."; return false; }
        }
        const size_t inputBytes = input.size() * sizeof(input.front()), bytes = input.size() * sizeof(uint32_t);
        if (std::max(inputBytes, bytes) > Device.maxBufferLength)
        { error = "Metal adaptive allocation exceeds the device buffer limit."; return false; }
        Buffer raw, counts;
        if (!CreateVectorBuffer(raw, input, "Arvo Metal allocation pilots", error) ||
            !CreateBuffer(counts, bytes, nullptr, "Arvo Metal production budgets", error)) return false;
        const auto pixels = uint32_t(input.size());
        for (controls.Begin = 0; controls.Begin < pixels; controls.Begin += Batch)
        {
            if (cancellation()) return false;
            controls.Count = std::min(Batch, pixels - controls.Begin);
            @autoreleasepool
            {
                id<MTLCommandBuffer> command = [CommandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (command == nil || encoder == nil) { error = "Metal allocation command creation failed."; return false; }
                [encoder setComputePipelineState:AdaptiveDenoisePipeline];
                [encoder setBuffer:raw.Handle offset:0 atIndex:0];
                [encoder setBuffer:counts.Handle offset:0 atIndex:1];
                [encoder setBytes:&controls length:sizeof(controls) atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake((controls.Count + Threads - 1) / Threads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(Threads, 1, 1)];
                [encoder endEncoding];
                if (!CommitAndWait(command, "Metal adaptive allocation", cancellation, error)) return false;
            }
        }
        if (cancellation()) return false;
        const auto* data = static_cast<const uint32_t*>(counts.Handle.contents);
        if (!data) { error = "Metal production budgets are not CPU-visible."; return false; }
        result.assign(data, data + pixels);
        return true;
    }

    GeometryGuideResult CaptureGuidesNative(const Scene &scene, const Camera &camera,
                                            const RenderSettings &settings,
                                            const Detail::CancellationProbe &stopped) const
    {
        return Detail::CaptureGeometryGuides(
            scene, camera, settings, stopped, [&](const auto &controls, auto &rays, auto &error)
            { return GeometryGuideStage(controls, {}, {}, stopped, rays, error); }, [&](const auto &rays)
            { return QueryNative(scene, rays, {settings.Acceleration, settings.BvhPolicy}, stopped); },
            [&](const auto &controls, const auto &rays, const auto &hits, auto &guides, auto &error)
            { return GeometryGuideStage(controls, rays, hits, stopped, guides, error); });
    }

    bool ReconstructNative(const LinearFilm &film, const RenderSettings &settings,
                           const GeometryGuideImage *guides, const Detail::CancellationProbe &cancellation,
                           ReconstructedImage &result, std::string &error) const
    {
        namespace Data = ReconstructionData;
        if (!ValidateReconstructionInput(film, settings.Reconstruction, error) ||
            !ValidateReconstructionGuides(film, settings.Reconstruction, guides, error) ||
            !EnsureSpatialDenoisePipeline(error))
            return false;
        const size_t inputBytes = film.Pixels.size() * sizeof(Data::Pixel);
        const size_t colorBytes = film.Pixels.size() * sizeof(Vec4);
        const size_t displayBytes = film.Pixels.size() * sizeof(uint32_t);
        if (std::max({inputBytes, colorBytes, displayBytes}) > Device.maxBufferLength)
        {
            error = "Metal reconstruction film exceeds the device's maximum buffer length.";
            return false;
        }
        std::vector<Data::Pixel> input;
        input.reserve(film.Pixels.size());
        for (const auto& pixel : film.Pixels)
        {
            if (cancellation())
                return false;
            input.push_back({pixel.MeanRadiance.X, pixel.MeanRadiance.Y, pixel.MeanRadiance.Z,
                             pixel.LuminanceM2, pixel.SampleCount});
            if (guides)
                Data::PackGuide(input.back(), guides->Pixels[input.size() - 1]);
        }
        Buffer raw, previous, next, display, failures;
        if (!CreateVectorBuffer(raw, input, "Arvo Metal reconstruction raw film", error) ||
            !CreateBuffer(previous, colorBytes, nullptr, "Arvo Metal reconstruction previous", error) ||
            !CreateBuffer(next, colorBytes, nullptr, "Arvo Metal reconstruction next", error) ||
            !CreateBuffer(display, displayBytes, nullptr, "Arvo Metal reconstruction display", error) ||
            !CreateBuffer(failures, sizeof(uint32_t), nullptr, "Arvo Metal reconstruction status", error))
            return false;
        if (!failures.Handle.contents)
        {
            error = "Metal reconstruction status is not CPU-visible.";
            return false;
        }
        std::memset(failures.Handle.contents, 0, sizeof(uint32_t));
        const auto& options = settings.Reconstruction;
        Data::Controls controls{film.Width,
                                film.Height,
                                0,
                                0,
                                0,
                                ReconstructionUsesMoments(settings.Integrator) ? Data::MomentGuidance : 0,
                                options.NoiseScale,
                                options.RelativeColorScale,
                                options.AbsoluteColorScale,
                                settings.GammaCorrection ? 1u : 0u,
                                options.NormalScale,
                                options.RelativePlaneScale};
        if (guides)
            controls.Flags |= Data::GeometryGuidance;
        constexpr uint32_t Threads = 64, Batch = 16384;
        const uint32_t pixels = uint32_t(film.Pixels.size());
        for (controls.Pass = 0; controls.Pass < options.Passes; ++controls.Pass)
        {
            for (controls.Begin = 0; controls.Begin < pixels; controls.Begin += Batch)
            {
                if (cancellation())
                    return false;
                controls.Count = std::min(Batch, pixels - controls.Begin);
                @autoreleasepool
                {
                    id<MTLCommandBuffer> command = [CommandQueue commandBuffer];
                    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                    if (command == nil || encoder == nil)
                    {
                        error = "Metal reconstruction command allocation failed.";
                        return false;
                    }
                    [encoder setComputePipelineState:SpatialDenoisePipeline];
                    [encoder setBuffer:raw.Handle offset:0 atIndex:0];
                    [encoder setBuffer:previous.Handle offset:0 atIndex:1];
                    [encoder setBuffer:next.Handle offset:0 atIndex:2];
                    [encoder setBuffer:failures.Handle offset:0 atIndex:3];
                    [encoder setBytes:&controls length:sizeof(controls) atIndex:4];
                    [encoder setBuffer:display.Handle offset:0 atIndex:5];
                    [encoder dispatchThreadgroups:MTLSizeMake((controls.Count + Threads - 1) / Threads, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(Threads, 1, 1)];
                    [encoder endEncoding];
                    if (!CommitAndWait(command, "Metal spatial denoising", cancellation, error))
                        return false;
                }
                uint32_t failure = 0;
                std::memcpy(&failure, failures.Handle.contents, sizeof(failure));
                if (failure)
                {
                    error = "Metal spatial denoising reported invalid state.";
                    return false;
                }
            }
            std::swap(previous, next);
        }
        const auto* colors = static_cast<const Vec4*>(previous.Handle.contents);
        const auto* rgba = static_cast<const uint8_t*>(display.Handle.contents);
        if (!colors || !rgba)
        {
            error = "Metal reconstructed image is not CPU-visible.";
            return false;
        }
        ReconstructedImage output{film.Width, film.Height, {}, Image(film.Width, film.Height)};
        output.Pixels.reserve(pixels);
        for (uint32_t row = 0; row < film.Height; ++row)
        {
            if (cancellation())
                return false;
            const size_t first = size_t(row) * film.Width;
            for (uint32_t column = 0; column < film.Width; ++column)
            {
                const Vec4 pixel = colors[first + column];
                output.Pixels.push_back({pixel.X, pixel.Y, pixel.Z});
            }
            std::memcpy(output.Display.GetPixelsRGBA8().data() + first * sizeof(uint32_t),
                        rgba + first * sizeof(uint32_t), size_t(film.Width) * sizeof(uint32_t));
        }
        result = std::move(output);
        return true;
    }

    bool EnsurePointGiPipelines(std::string& error) const
    {
        namespace G = MetalPointGi;
        if (PointGiPipelines[G::Initialize] != nil)
            return true;
        if (@available(macOS 11.0, *))
        {
            NSError* detail = nil;
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            options.languageVersion = MTLLanguageVersion2_3;
            options.fastMathEnabled = NO;
            id<MTLLibrary> library =
                [Device newLibraryWithSource:[NSString stringWithUTF8String:Detail::MetalPointGiSource]
                                     options:options
                                       error:&detail];
            ReleaseObject(options);
            if (library == nil)
            {
                error = ErrorDescription(detail, "Metal Point GI compilation failed.");
                return false;
            }
            constexpr const char* Entries[G::StageCount] = {"ArvoPointGiInitialize", "ArvoPointGiBake",
                                                            "ArvoPointGiCamera"};
            id<MTLComputePipelineState> compiled[G::StageCount] = {};
            bool valid = true;
            for (uint32_t stage = 0; stage < G::StageCount && valid; ++stage)
            {
                id<MTLFunction> entry =
                    [library newFunctionWithName:[NSString stringWithUTF8String:Entries[stage]]];
                if (entry != nil)
                    compiled[stage] = [Device newComputePipelineStateWithFunction:entry error:&detail];
                ReleaseObject(entry);
                if (compiled[stage] == nil || compiled[stage].maxTotalThreadsPerThreadgroup < G::Threads)
                {
                    error =
                        std::string(Entries[stage]) + ": " +
                        ErrorDescription(detail, "Metal Point GI pipeline cannot support its threadgroup.");
                    valid = false;
                }
            }
            ReleaseObject(library);
            for (uint32_t stage = 0; stage < G::StageCount; ++stage)
                if (valid)
                    PointGiPipelines[stage] = compiled[stage];
                else
                    ReleaseObject(compiled[stage]);
            return valid;
        }
        error = "Metal Point GI requires macOS 11 or later for MSL 2.3 arithmetic.";
        return false;
    }

    bool RenderPointGi(const Scene& scene, const FrameConstants& constants, const PointGiOptions& options,
                       const Detail::CancellationProbe& cancellation, RenderResult& result) const
    {
        namespace G = MetalPointGi;
        if (!ValidatePointGiOptions(options, result.Error))
            return false;
        const auto emission = MetalEmission::PrepareEmitters(scene);
        constexpr size_t UniformSourceLimit = size_t(1) << 24;
        if (emission.Sources.size() > UniformSourceLimit)
        {
            result.Error = "Metal Point GI emitter count exceeds the uniform sampler's exact range.";
            return false;
        }
        for (const auto& source : emission.Sources)
            if (!(source.Flags & EmitterCastsShadows))
            {
                result.Error = "Metal Point GI requires shadow casting on every emitter.";
                return false;
            }
        G::Geometry geometry;
        if (constants.MaxReflectionBounces &&
            !G::BuildGeometry(scene, options, cancellation, geometry, result.Error))
            return false;
        if (cancellation() || !EnsurePointGiPipelines(result.Error))
            return false;
        const size_t radianceBytes = geometry.Surfaces.size() * sizeof(G::Radiance);
        const size_t maximumBytes = std::max(
            {geometry.Surfaces.size() * sizeof(G::Surface), geometry.Nodes.size() * sizeof(G::Node),
             geometry.Indices.size() * sizeof(uint32_t), emission.Sources.size() * sizeof(PackedEmitter),
             radianceBytes, size_t(G::WorkBatch) * sizeof(GroupStatistics)});
        if (maximumBytes > Device.maxBufferLength)
        {
            result.Error = "Metal Point GI cache exceeds the device's maximum buffer length.";
            return false;
        }
        Buffer sources, surfaces, nodes, indices, previous, next, total, statistics, errors;
        if (!CreateVectorBuffer(sources, emission.Sources, "Arvo Metal Point GI emitters", result.Error) ||
            !CreateVectorBuffer(surfaces, geometry.Surfaces, "Arvo Metal Point GI surfaces", result.Error) ||
            !CreateVectorBuffer(nodes, geometry.Nodes, "Arvo Metal Point GI hierarchy", result.Error) ||
            !CreateVectorBuffer(indices, geometry.Indices, "Arvo Metal Point GI indices", result.Error) ||
            !CreateBuffer(previous, radianceBytes, nullptr, "Arvo Metal Point GI previous order",
                          result.Error) ||
            !CreateBuffer(next, radianceBytes, nullptr, "Arvo Metal Point GI next order", result.Error) ||
            !CreateBuffer(total, radianceBytes, nullptr, "Arvo Metal Point GI total", result.Error) ||
            !CreateBuffer(statistics, size_t(G::WorkBatch) * sizeof(GroupStatistics), nullptr,
                          "Arvo Metal Point GI statistics", result.Error) ||
            !CreateBuffer(errors, sizeof(uint32_t), nullptr, "Arvo Metal Point GI status", result.Error))
            return false;
        G::Controls controls{uint32_t(geometry.Surfaces.size()),
                             uint32_t(geometry.Nodes.size()),
                             uint32_t(emission.Sources.size()),
                             options.Seed ^ constants.SamplingSeed,
                             options.DirectSamples,
                             options.GatherSamples};
        const auto dispatch = [&](uint32_t stage, uint32_t first, uint32_t count) {
            if (cancellation() || !ResetTraversalError(result.Error))
                return false;
            controls.Begin = first;
            controls.Count = count;
            @autoreleasepool
            {
                id<MTLCommandBuffer> command = [CommandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (command == nil || encoder == nil)
                {
                    result.Error = "Metal Point GI command allocation failed.";
                    return false;
                }
                [encoder setComputePipelineState:PointGiPipelines[stage]];
                [encoder setBuffer:Nodes.Handle offset:0 atIndex:0];
                [encoder setBuffer:PrimitiveRefs.Handle offset:0 atIndex:1];
                [encoder setBuffer:Triangles.Handle offset:0 atIndex:2];
                [encoder setBuffer:Spheres.Handle offset:0 atIndex:3];
                [encoder setBuffer:Materials.Handle offset:0 atIndex:4];
                [encoder setBuffer:sources.Handle offset:0 atIndex:5];
                [encoder setBytes:&constants length:sizeof(constants) atIndex:6];
                [encoder setBuffer:Accumulation.Handle offset:0 atIndex:7];
                [encoder setBuffer:SampleCounts.Handle offset:0 atIndex:8];
                [encoder setBuffer:statistics.Handle offset:0 atIndex:9];
                [encoder setBuffer:TextureDescriptors.Handle offset:0 atIndex:10];
                [encoder setBuffer:TextureTexels.Handle offset:0 atIndex:11];
                [encoder setBuffer:SampleTargets.Handle offset:0 atIndex:12];
                BindInstanceBuffers(encoder);
                [encoder setBuffer:surfaces.Handle offset:0 atIndex:G::SurfacesBinding];
                [encoder setBuffer:nodes.Handle offset:0 atIndex:G::NodesBinding];
                [encoder setBuffer:indices.Handle offset:0 atIndex:G::IndicesBinding];
                [encoder setBuffer:previous.Handle offset:0 atIndex:G::PreviousBinding];
                [encoder setBuffer:next.Handle offset:0 atIndex:G::NextBinding];
                [encoder setBuffer:total.Handle offset:0 atIndex:G::TotalBinding];
                [encoder setBytes:&controls length:sizeof(controls) atIndex:G::ControlsBinding];
                [encoder setBuffer:errors.Handle offset:0 atIndex:G::ErrorsBinding];
                const uint32_t groups = (count + G::Threads - 1) / G::Threads;
                [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(G::Threads, 1, 1)];
                [encoder endEncoding];
                // Each order becomes immutable before swapping buffers. The command
                // retains its resources if the shared device-lost timeout fires.
                if (!CommitAndWait(command, "Metal Point GI dispatch", cancellation, result.Error) ||
                    !CheckTraversalError(result.Error))
                    return false;
            }
            const auto* failure = static_cast<const uint32_t*>(errors.Handle.contents);
            if (!failure || *failure)
            {
                result.Error =
                    "Metal Point GI kernel reported invalid configuration, cache or numerical state.";
                return false;
            }
            if (stage != G::Initialize)
            {
                const auto* counters = static_cast<const GroupStatistics*>(statistics.Handle.contents);
                if (!counters)
                {
                    result.Error = "Metal Point GI statistics are not CPU-visible.";
                    return false;
                }
                for (uint32_t row = 0; row < count; ++row)
                {
                    result.Statistics.PrimaryRays += counters[row].PrimaryRays.Value();
                    result.Statistics.ShadowRays += counters[row].ShadowRays.Value();
                    result.Statistics.ReflectionRays += counters[row].ReflectionRays.Value();
                    result.Statistics.PrimitiveTests += counters[row].PrimitiveTests.Value();
                }
            }
            return true;
        };
        const auto population = [&](uint32_t stage, uint32_t count) {
            for (uint32_t first = 0; first < count; first += G::WorkBatch)
                if (!dispatch(stage, first, std::min(G::WorkBatch, count - first)))
                    return false;
            return true;
        };
        const auto samples = [&](uint32_t stage, uint32_t count, uint32_t budget) {
            for (controls.FirstSample = 0; controls.FirstSample < budget;
                 controls.FirstSample += G::SampleBatch)
            {
                controls.Samples = std::min(G::SampleBatch, budget - controls.FirstSample);
                if (!population(stage, count))
                    return false;
            }
            return true;
        };
        const uint32_t pixels = constants.Width * constants.Height;
        if (!population(G::Initialize, pixels))
            return false;
        for (controls.Order = 0; controls.Order < constants.MaxReflectionBounces; ++controls.Order)
        {
            if (!samples(G::Bake, controls.Surfaces,
                         controls.Order == 0 ? controls.DirectSamples : controls.GatherSamples))
                return false;
            std::swap(previous, next);
        }
        return samples(G::Camera, pixels, constants.TotalSamples);
    }

    [[nodiscard]] RenderResult RenderNative(
        const Scene& scene,
        const std::shared_ptr<const Scene>& owner,
        const Camera& camera,
        const RenderSettings& settings,
        const Detail::CancellationProbe& cancellation) const
    {
        RenderResult result;
        if (!ValidateInput(scene, camera, settings, result.Error))
        {
            return result;
        }
        if (!Detail::ValidateMaterialTransport(scene, BackendType::Metal,
                settings.Integrator, result.Error)) return result;
        const bool bidirectional = settings.Integrator == Algorithm::BidirectionalPathTracing;
        const bool metropolis = settings.Integrator == Algorithm::MetropolisLightTransport;
        const bool photon = settings.Integrator == Algorithm::PhotonMapping;
        const bool progressive = settings.Integrator == Algorithm::ProgressivePhotonMapping;
        const bool vcm = settings.Integrator == Algorithm::VertexConnectionMerging;
        const bool pointGi = settings.Integrator == Algorithm::PointBasedGlobalIllumination;
        if (pointGi && (settings.Adaptive.Enabled || settings.Transport.RussianRoulette ||
            settings.MaxReflectionBounces > MetalPointGi::MaximumReflections ||
            settings.Transport.Mis != MisHeuristic::None || !settings.Transport.NextEventEstimation ||
            !settings.ShadowsEnabled || settings.ShadowStrength != 1))
            return Detail::MakeFailedResult("Metal Point GI requires fixed samples/depth, NEE, no MIS and full visibility.");
        if (vcm && (settings.Adaptive.Enabled || settings.Transport.RussianRoulette ||
            !settings.ShadowsEnabled || settings.ShadowStrength != 1))
            return Detail::MakeFailedResult("Metal VCM requires fixed passes/depth and full physical visibility.");
        if (progressive && (settings.Adaptive.Enabled || settings.MaxReflectionBounces > MetalPpm::MaximumReflections ||
            settings.Transport.Mis != MisHeuristic::None || !settings.ShadowsEnabled || settings.ShadowStrength != 1))
            return Detail::MakeFailedResult("Metal PPM requires fixed passes, at most 32 reflections, no MIS and full physical visibility.");
        if (photon && (settings.Adaptive.Enabled || settings.MaxReflectionBounces > MetalPhoton::MaximumReflections ||
            settings.Transport.Mis != MisHeuristic::None || !settings.ShadowsEnabled || settings.ShadowStrength != 1))
            return Detail::MakeFailedResult("Metal photon mapping requires fixed samples, at most 32 reflections, no connection MIS and full physical visibility.");
        if (bidirectional && (settings.Adaptive.Enabled || settings.Transport.RussianRoulette ||
            !settings.Transport.NextEventEstimation || !settings.ShadowsEnabled || settings.ShadowStrength != 1))
            return Detail::MakeFailedResult("Metal BDPT requires fixed samples/depth, NEE and full physical visibility.");
        if (metropolis && (settings.Adaptive.Enabled || settings.MaxReflectionBounces > MetalMlt::MaximumReflections ||
            !settings.ShadowsEnabled || settings.ShadowStrength != 1))
            return Detail::MakeFailedResult("Metal MLT requires fixed samples, at most 32 reflections and full physical visibility.");
        const auto buildBegin = std::chrono::steady_clock::now();
        if (!EnsureScene(scene, owner, {settings.Acceleration, settings.BvhPolicy}, cancellation,
                         result.Error,
                         settings.Radiometry == RadiometryMode::Spectral ? &settings.Medium : nullptr))
        {
            if (cancellation())
            {
                return Detail::MakeCancelledResult();
            }
            return result;
        }
        const auto buildEnd = std::chrono::steady_clock::now();
        if (cancellation())
        {
            return Detail::MakeCancelledResult();
        }

        const std::uint32_t pixelCount = settings.Width * settings.Height;
        if (!vcm && (!EnsureFrameBuffers(pixelCount, result.Error)
            || !ClearFrame(pixelCount, cancellation, result.Error)))
        {
            if (cancellation())
            {
                return Detail::MakeCancelledResult();
            }
            return result;
        }

        FrameConstants constants;
        constants.CameraPosition = ToVec3(camera.Position);
        constants.Width = settings.Width;
        constants.FullWidth = FullImageWidth(settings);
        constants.FullHeight = FullImageHeight(settings);
        constants.TileX = settings.TileX;
        constants.TileY = settings.TileY;
        constants.Integrator = static_cast<std::uint32_t>(settings.Integrator);
        constexpr uint32_t SpectralTransportBit = 1u << 5;
        constants.TransportFlags =
            PackTransportFlags(settings.Transport) |
            (settings.Radiometry == RadiometryMode::Spectral ? SpectralTransportBit : 0);
        constants.RouletteStartDepth = settings.Transport.RouletteStartDepth;
        constants.SamplingSequence = static_cast<uint32_t>(settings.Sampling);
        constants.SamplingSeed = settings.SamplingSeed;
        constants.Acceleration = static_cast<std::uint32_t>(settings.Acceleration);
        constants.AdaptiveEnabled = settings.Adaptive.Enabled ? 1u : 0u;
        constants.AdaptivePilotSamples = settings.Adaptive.PilotSamples;
        constants.AdaptiveMinimumSamples = settings.Adaptive.MinimumSamples;
        constants.AdaptiveRelativeError = settings.Adaptive.RelativeError;
        constants.AdaptiveAbsoluteError = settings.Adaptive.AbsoluteError;
        constants.Height = settings.Height;
        constants.TotalSamples = settings.SamplesPerPixel;
        constants.MaxReflectionBounces = settings.MaxReflectionBounces;
        const Vec3 forward = NormalizeVector(
            Subtract(ToVec3(camera.Target), ToVec3(camera.Position)));
        const Vec3 right = NormalizeVector(CrossVector(forward, ToVec3(camera.Up)));
        constants.CameraForward = forward;
        constants.LensRadius = camera.Lens.Radius;
        constants.FocusDistance = camera.Lens.FocusDistance;
        constants.CameraRight = right;
        constants.CameraUp = NormalizeVector(CrossVector(right, forward));
        constants.HalfHeight =
            std::tan(camera.VerticalFieldOfViewDegrees * Pi / 360.0f);
        constants.HalfWidth = constants.HalfHeight
            * static_cast<float>(FullImageWidth(settings))
            / static_cast<float>(FullImageHeight(settings));
        constants.MaterialCount = MaterialCount;
        constants.LightCount = LightCount;
        constants.BackgroundTop = ToVec3(scene.BackgroundTop);
        constants.AmbientIntensity = scene.AmbientIntensity;
        constants.BackgroundBottom = ToVec3(scene.BackgroundBottom);
        constants.NodeCount = NodeCount;
        constants.InstanceNodeCount = InstanceNodeCount;
        constants.PrimitiveRefCount = PrimitiveRefCount;
        constants.TriangleCount = TriangleCount;
        constants.SphereCount = SphereCount;
        constants.ShadowStrength = settings.ShadowsEnabled
            ? std::clamp(settings.ShadowStrength, 0.0f, 1.0f)
            : 0.0f;
        constants.GammaCorrection = settings.GammaCorrection ? 1u : 0u;
        constants.TextureCount = TextureCount;
        constants.TextureTexelCount = TextureTexelCount;

        const auto renderBegin = std::chrono::steady_clock::now();
        if (vcm)
        {
            bool success = RenderVcm(scene, constants, settings, cancellation, result);
            result.Statistics.BuildMilliseconds = std::chrono::duration<double, std::milli>(buildEnd - buildBegin).count();
            result.Statistics.RenderMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - renderBegin).count();
            result.Status = cancellation() ? RenderStatus::Cancelled : success ? RenderStatus::Completed : RenderStatus::Failed;
            if (!success) { result.Film = {}; result.Output = {}; result.Statistics.CompletedPixels = 0; }
            return result;
        }
        constexpr std::uint32_t PilotPhase = 1u;
        const bool variancePilot = settings.Adaptive.Enabled && !settings.ProductionBudgets;
        const std::uint32_t phaseCount = variancePilot ? 2u : 1u;
        if (settings.ProductionBudgets)
            std::memcpy(SampleTargets.Handle.contents, settings.ProductionBudgets->Counts.data(),
                        size_t(pixelCount) * sizeof(uint32_t));
        if (bidirectional && !RenderBidirectional(scene, constants, cancellation, result))
        {
            if (cancellation()) result.Status = RenderStatus::Cancelled;
            return result;
        }
        if (metropolis && !RenderMetropolis(scene, constants, settings.Metropolis, cancellation, result))
        {
            if (cancellation()) result.Status = RenderStatus::Cancelled;
            return result;
        }
        if (photon && !RenderPhotons(scene, constants, settings.Photons, cancellation, result))
        {
            if (cancellation()) result.Status = RenderStatus::Cancelled;
            return result;
        }
        if (progressive && !RenderProgressivePhotons(scene, constants, settings.ProgressivePhotons, cancellation, result))
        {
            if (cancellation()) result.Status = RenderStatus::Cancelled;
            return result;
        }
        if (pointGi && !RenderPointGi(scene, constants, settings.PointGi, cancellation, result))
        {
            if (cancellation()) result.Status = RenderStatus::Cancelled;
            return result;
        }
        for (std::uint32_t phase = 0u; !bidirectional && !metropolis && !photon && !progressive && !pointGi && phase < phaseCount; ++phase)
        {
            const bool pilot = variancePilot && phase == 0u;
            constants.AdaptivePhase = pilot ? PilotPhase : uint32_t(settings.SampleStream);
            const std::uint32_t phaseSamples = pilot ? settings.Adaptive.PilotSamples : settings.SamplesPerPixel;
            for (std::uint32_t sample = 0u;
                 sample < phaseSamples;
                 ++sample)
            {
                std::uint64_t passPrimaryRays = 0;
                for (std::uint32_t begin = 0u;
                     begin < pixelCount;
                     begin += PixelChunkSize)
                {
                    if (cancellation())
                    {
                        result.Status = RenderStatus::Cancelled;
                        result.Statistics.BuildMilliseconds =
                            std::chrono::duration<double, std::milli>(
                                buildEnd - buildBegin).count();
                        result.Statistics.RenderMilliseconds =
                            std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - renderBegin).count();
                        return result;
                    }
                    constants.PixelBegin = begin;
                    constants.PixelCount = std::min(PixelChunkSize, pixelCount - begin);
                    constants.SampleBegin = sample;
                    constants.SampleCount = 1u;
                    constants.GroupCount =
                        (constants.PixelCount + ThreadsPerGroup - 1u)
                        / ThreadsPerGroup;
                    GroupStatistics statistics;
                    if (!DispatchTrace(constants, PackMedium(settings.Medium), {camera.Motion, HasSceneMotion(scene)}, statistics,
                                       cancellation, result.Error))
                    {
                        if (cancellation())
                        {
                            result.Status = RenderStatus::Cancelled;
                        }
                        return result;
                    }
                    result.Statistics.PrimaryRays += statistics.PrimaryRays.Value();
                    passPrimaryRays += statistics.PrimaryRays.Value();
                    if (pilot) result.Statistics.PilotRays += statistics.PrimaryRays.Value();
                    result.Statistics.ShadowRays += statistics.ShadowRays.Value();
                    result.Statistics.ReflectionRays +=
                        statistics.ReflectionRays.Value();
                    result.Statistics.PrimitiveTests +=
                        statistics.PrimitiveTests.Value();
                }
                // Fixed targets cannot become active again after an empty pass.
                if (!pilot && passPrimaryRays == 0) break;
            }
        }

        if (!ResolvePixels(constants, pixelCount, cancellation, result.Error))
        {
            if (cancellation())
            {
                result.Status = RenderStatus::Cancelled;
            }
            return result;
        }
        if (cancellation())
        {
            return Detail::MakeCancelledResult();
        }

        const void* output = Output.Handle.contents;
        const auto* completedSamples =
            static_cast<const std::uint32_t*>(SampleCounts.Handle.contents);
        const auto* filmMoments = static_cast<const Vec4*>(Accumulation.Handle.contents);
        const auto* sampleTargets = static_cast<const std::uint32_t*>(SampleTargets.Handle.contents);
        if (output == nullptr || completedSamples == nullptr || filmMoments == nullptr || sampleTargets == nullptr)
        {
            result.Error = "Metal frame readback buffers are not CPU-visible.";
            return result;
        }
        result.Output = Image(settings.Width, settings.Height);
        result.Film = LinearFilm(settings.Width, settings.Height);
        std::vector<std::uint8_t>& pixels = result.Output.GetPixelsRGBA8();
        constexpr std::size_t ReadbackChunkSize = 32u * 1024u * 1024u;
        const std::size_t outputBytes =
            static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t);
        for (std::size_t offset = 0u;
             offset < outputBytes;
             offset += ReadbackChunkSize)
        {
            if (cancellation())
            {
                return Detail::MakeCancelledResult();
            }
            const std::size_t chunk =
                std::min(ReadbackChunkSize, outputBytes - offset);
            std::memcpy(
                pixels.data() + offset,
                static_cast<const std::uint8_t*>(output) + offset,
                chunk);
        }
        for (std::uint32_t pixel = 0u; pixel < pixelCount; ++pixel)
        {
            const std::uint32_t minimum = settings.Adaptive.Enabled ? settings.Adaptive.MinimumSamples : settings.SamplesPerPixel;
            if (sampleTargets[pixel] < minimum || sampleTargets[pixel] > settings.SamplesPerPixel
                || completedSamples[pixel] != sampleTargets[pixel])
            {
                result.Error = "Metal did not complete the selected per-pixel sample budget.";
                return result;
            }
            const Vec4& value = filmMoments[pixel];
            result.Film.Pixels[pixel] = {Float3(value.X, value.Y, value.Z), value.W, completedSamples[pixel]};
            ++result.Statistics.CompletedPixels;
        }
        const auto renderEnd = std::chrono::steady_clock::now();
        result.Statistics.BuildMilliseconds =
            std::chrono::duration<double, std::milli>(
                buildEnd - buildBegin).count();
        result.Statistics.RenderMilliseconds =
            std::chrono::duration<double, std::milli>(
                renderEnd - renderBegin).count();
        result.Status = cancellation()
            ? RenderStatus::Cancelled
            : RenderStatus::Completed;
        return result;
    }

    id<MTLDevice> Device = nil;
    id<MTLCommandQueue> CommandQueue = nil;
    id<MTLComputePipelineState> WhittedPipeline = nil;
    id<MTLComputePipelineState> QueryPipeline = nil;
    id<MTLComputePipelineState> PathTracingPipeline = nil;
    id<MTLComputePipelineState> DistributedRayPipeline = nil;
    id<MTLComputePipelineState> ResolvePipeline = nil;
    mutable id<MTLComputePipelineState> SpatialDenoisePipeline = nil;
    mutable id<MTLComputePipelineState> AdaptiveDenoisePipeline = nil;
    mutable id<MTLComputePipelineState> GeometryGuidePipelines[2] = {};
    mutable id<MTLComputePipelineState> TemporalDenoisePipeline = nil;
    mutable id<MTLComputePipelineState> DicingPipeline = nil;
    mutable id<MTLComputePipelineState> BidirectionalPairs = nil;
    mutable id<MTLComputePipelineState> BidirectionalClear = nil;
    mutable id<MTLComputePipelineState> BidirectionalCommit = nil;
    mutable id<MTLComputePipelineState> MetropolisPipelines[MetalMlt::StageCount] = {};
    mutable id<MTLComputePipelineState> PhotonPipelines[MetalPhoton::StageCount] = {};
    mutable id<MTLComputePipelineState> ProgressivePipelines[MetalPpm::StageCount] = {};
    mutable id<MTLComputePipelineState> PointGiPipelines[MetalPointGi::StageCount] = {};
    mutable id<MTLComputePipelineState> VcmPipelines[MetalVcm::StageCount] = {};
    mutable std::mutex AvailabilityMutex;
    mutable bool Available = false;
    mutable bool Terminal = false;
    mutable std::string AvailabilityError;
    mutable std::string DeviceName;

    mutable Buffer Nodes;
    mutable Buffer InstanceRecords;
    mutable Buffer InstanceNodes;
    mutable Buffer InstanceError;
    mutable Buffer PrimitiveRefs;
    mutable Buffer Triangles;
    mutable Buffer Spheres;
    mutable Buffer Materials;
    mutable Buffer Lights;
    mutable Buffer TextureDescriptors;
    mutable Buffer TextureTexels;
    mutable Buffer SpectralPalette;
    mutable Buffer SpectralReferences;
    mutable Buffer SpectralFallback;
    mutable uint32_t SpectralSceneSourceCount = 0;
    mutable MediumOptions SpectralMedium;
    mutable id<MTLComputePipelineState> SpectralResponsePipeline = nil, SpectralPalettePipeline = nil;
    mutable Buffer Accumulation;
    mutable Buffer SampleCounts;
    mutable Buffer SampleTargets;
    mutable Buffer Output;
    mutable Buffer GroupStats;
    mutable std::uint32_t NodeCount = 0u;
    mutable std::uint32_t InstanceNodeCount = 0u;
    mutable std::uint32_t PrimitiveRefCount = 0u;
    mutable std::uint32_t TriangleCount = 0u;
    mutable std::uint32_t SphereCount = 0u;
    mutable std::uint32_t MaterialCount = 0u;
    mutable std::uint32_t LightCount = 0u;
    mutable std::uint32_t TextureCount = 0u;
    mutable std::uint32_t TextureTexelCount = 0u;
    mutable std::uint32_t FramePixelCapacity = 0u;
    mutable std::weak_ptr<const Scene> CachedOwner;
    mutable AccelerationOptions CachedAcceleration = AccelerationType::Bvh;
    mutable Detail::PersistentRenderQueue MetalQueue;
};

MetalRayTracingRHI::MetalRayTracingRHI()
    : m_Impl(std::make_unique<Impl>())
{
}

MetalRayTracingRHI::~MetalRayTracingRHI() = default;

BackendInfo MetalRayTracingRHI::GetBackendInfo() const
{
    const Impl::AvailabilitySnapshot availability = m_Impl->GetAvailability();
    BackendInfo info;
    info.Type = BackendType::Metal;
    info.Name = "Metal Compute";
    info.IsNative = availability.Available;
    info.IsHardwareAccelerated = availability.Available;
    info.IsAvailable = availability.Available;
    info.Description = availability.Available
        ? std::string("Native Metal compute ray tracing backend on ")
            + availability.DeviceName
        : std::string("Unavailable")
            + (availability.Terminal ? " (terminal)" : "")
            + ": "
            + availability.Error;
    return info;
}

RayQueryResult MetalRayTracingRHI::QueryRays(const Scene& scene, const std::vector<Ray>& rays,
    AccelerationOptions acceleration,
    const CancellationToken& cancellation) const
{
    RayQueryResult result;
    // Use the same queue as renders so packed scene and GPU resources are serial.
    auto job = m_Impl->MetalQueue.Submit([&](const CancellationToken& queued)
    {
        const auto probe = Detail::CombineCancellation(queued, cancellation);
        if (probe()) return Detail::MakeCancelledResult();
        const auto availability = m_Impl->GetAvailability();
        if (!availability.Available) return Detail::MakeFailedResult("Metal is unavailable: " + availability.Error);
        @autoreleasepool
        {
            result = m_Impl->QueryNative(scene, rays, acceleration, probe);
            RenderResult completion;
            completion.Status = result.Status;
            completion.Error = result.Error;
            return completion;
        }
    });
    const auto completion = job.GetResult();
    result.Status = completion.Status;
    result.Error = completion.Error;
    return result;
}

MicropolygonResult MetalRayTracingRHI::PrepareMicropolygons(const Scene& scene, const Camera& camera,
                                                            const RenderSettings& settings,
                                                            const CancellationToken& cancellation) const
{
    MicropolygonResult result;
    const auto task =
        m_Impl->MetalQueue
            .Submit([&](const CancellationToken& queued) {
                const auto stopped = Detail::CombineCancellation(queued, cancellation);
                if (stopped())
                    return Detail::MakeCancelledResult();
                const auto availability = m_Impl->GetAvailability();
                if (!availability.Available)
                    return Detail::MakeFailedResult("Metal is unavailable: " + availability.Error);
                @autoreleasepool
                {
                    result = Detail::PrepareMicropolygons(scene, camera, settings, stopped,
                                                          [&](const auto& controls, const auto& patches,
                                                              const auto& ranges, auto& words, auto& error) {
                                                              return m_Impl->MicropolygonPassNative(
                                                                  controls, patches, ranges, words, stopped,
                                                                  error);
                                                          });
                    RenderResult completion;
                    completion.Status = result.Status;
                    completion.Error = result.Error;
                    return completion;
                }
            })
            .GetResult();
    result.Status = task.Status;
    result.Error = task.Error;
    return result;
}

TemporalResult MetalRayTracingRHI::ReconstructTemporal(const TemporalFrame& frame,
                                                       const TemporalHistory* history,
                                                       const TemporalOptions& options,
                                                       const CancellationToken& cancellation) const
{
    TemporalResult result;
    const auto task =
        m_Impl->MetalQueue
            .Submit([&](const CancellationToken& queued) {
                const auto stopped = Detail::CombineCancellation(queued, cancellation);
                if (stopped())
                    return Detail::MakeCancelledResult();
                const auto availability = m_Impl->GetAvailability();
                if (!availability.Available)
                    return Detail::MakeFailedResult("Metal is unavailable: " + availability.Error);
                @autoreleasepool
                {
                    result = Detail::ReconstructTemporal(
                        BackendType::Metal, frame, history, options, stopped,
                        [&](const auto& controls, const auto& current, const auto& guides,
                            const auto& previous, const auto& oldGuides, auto& output, auto& display,
                            auto& error) {
                            return m_Impl->ReconstructTemporalNative(controls, current, guides, previous,
                                                                     oldGuides, stopped, output, display,
                                                                     error);
                        });
                    RenderResult completion;
                    completion.Status = result.Status;
                    completion.Error = result.Error;
                    return completion;
                }
            })
            .GetResult();
    result.Status = task.Status;
    result.Error = task.Error;
    if (result.Status != RenderStatus::Completed)
    {
        result.History = {};
        result.Display = {};
        result.AcceptedHistoryPixels = 0;
    }
    return result;
}

GeometryGuideResult MetalRayTracingRHI::CaptureGeometryGuides(const Scene& scene, const Camera& camera,
                                                              const RenderSettings& settings,
                                                              const CancellationToken& cancellation) const
{
    GeometryGuideResult result;
    if (settings.Micropolygons.Enabled)
    {
        auto diced = PrepareMicropolygons(scene, camera, settings, cancellation);
        if (diced.Status != RenderStatus::Completed)
            return {diced.Status, {}, diced.Error};
        auto prepared = settings;
        prepared.Micropolygons.Enabled = false;
        prepared.DicingPlan.reset();
        return CaptureGeometryGuides(*diced.Geometry, camera, prepared, cancellation);
    }
    const auto task =
        m_Impl->MetalQueue
            .Submit([&](const CancellationToken& queued) {
                const auto stopped = Detail::CombineCancellation(queued, cancellation);
                if (stopped())
                    return Detail::MakeCancelledResult();
                const auto availability = m_Impl->GetAvailability();
                if (!availability.Available)
                    return Detail::MakeFailedResult("Metal is unavailable: " + availability.Error);
                @autoreleasepool
                {
                    result = m_Impl->CaptureGuidesNative(scene, camera, settings, stopped);
                    RenderResult completion;
                    completion.Status = result.Status;
                    completion.Error = result.Error;
                    return completion;
                }
            })
            .GetResult();
    result.Status = task.Status;
    result.Error = task.Error;
    if (result.Status != RenderStatus::Completed)
        result.Image = {};
    return result;
}

RenderResult MetalRayTracingRHI::Render(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings,
    const CancellationToken& cancellation) const
{
    return m_Impl->Submit(
        &scene,
        nullptr,
        camera,
        settings,
        cancellation).GetResult();
}

RenderJob MetalRayTracingRHI::RenderAsync(
    Scene scene,
    Camera camera,
    RenderSettings settings) const
{
    return RenderAsync(
        std::make_shared<Scene>(std::move(scene)),
        camera,
        settings);
}

RenderJob MetalRayTracingRHI::RenderAsync(
    std::shared_ptr<const Scene> scene,
    Camera camera,
    RenderSettings settings) const
{
    const Scene* pointer = scene.get();
    return m_Impl->Submit(pointer, std::move(scene), camera, settings);
}

void MetalRayTracingRHI::WaitIdle() const
{
    m_Impl->MetalQueue.WaitIdle();
}

} // namespace RayTracing
} // namespace Arvo

#endif
