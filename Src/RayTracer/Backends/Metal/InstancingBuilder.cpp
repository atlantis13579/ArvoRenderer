#include "MotionPreparation.h"
#include "InstancingBuilder.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace Arvo::RayTracing::MetalInstances
{
namespace
{
// Independent Metal TLAS: partition instance centroids along their greatest
// variance, then median-split into a depth-first tree with escape links. Object
// geometry is borrowed once per pointer identity, never expanded into vertices.
// PBRT, 4th ed., "Transformed Primitives" and "Bounding Volume Hierarchies".
// https://pbr-book.org/4ed/Primitives_and_Intersection_Acceleration
// The host inverse uses scaled, pivoted Gauss-Jordan elimination. Each entry is
// stored as (High + Low) * 2^Exponent, retaining roughly 48 significant bits
// and a separate exponent without requiring an MSL double storage type.
// Instancing.metal consumes this layout without native double operations.
constexpr uint32_t Dimensions = 3, AffineStride = Dimensions + 1;
constexpr uint32_t AugmentedColumns = Dimensions * 2, CornerCount = 1u << Dimensions;
constexpr uint32_t LeafCapacity = 4, MaximumDepth = 48, CancellationStride = 1024;
constexpr uint64_t LocalPrimitiveLimit = (uint64_t{1} << 31) - 1;
constexpr uint64_t MaximumRecords = std::numeric_limits<uint32_t>::max() / 2;
constexpr double FloatLimit = std::numeric_limits<float>::max();
constexpr double BoundRoundoff = (Dimensions * 2 + 2) * std::numeric_limits<double>::epsilon();
using Point = std::array<double, Dimensions>;
using Matrix = std::array<double, Dimensions * AffineStride>;

struct Bounds
{
    Point Low{INFINITY, INFINITY, INFINITY}, High{-INFINITY, -INFINITY, -INFINITY};
    void Include(const Point &point)
    {
        for (uint32_t axis = 0; axis < Dimensions; ++axis)
        {
            Low[axis] = std::min(Low[axis], point[axis]);
            High[axis] = std::max(High[axis], point[axis]);
        }
    }
};
struct ObjectInfo
{
    Bounds Box;
    uint32_t Index = 0;
};

uint64_t PrimitiveCount(const SceneGeometry &geometry)
{
    return uint64_t(geometry.Triangles.size()) + geometry.Spheres.size();
}

bool Singular(const std::array<float, Dimensions * AffineStride> &matrix)
{
    // Row scaling can turn an exactly singular input into a tiny nonzero pivot.
    // Sum the six determinant products as an error-free expansion first. A
    // product of two input floats is exact in double; FMA recovers the rounding
    // residual of the third factor. All possible float products fit in double.
    // Shewchuk (1997), "Adaptive Precision Floating-Point Arithmetic and Fast
    // Robust Geometric Predicates": https://www.cs.cmu.edu/~quake/robust.html
    constexpr uint32_t ProductTerms = Dimensions * 2, ExpansionCapacity = ProductTerms * 2;
    std::array<double, ExpansionCapacity> expansion{};
    uint32_t length = 0;
    const auto add = [&](double value) {
        uint32_t written = 0;
        for (uint32_t index = 0; index < length; ++index)
        {
            const double entry = expansion[index], sum = value + entry;
            const double recovered = sum - value;
            const double residual = (value - (sum - recovered)) + (entry - recovered);
            if (residual != 0)
                expansion[written++] = residual;
            value = sum;
        }
        if (value != 0)
            expansion[written++] = value;
        length = written;
    };
    for (uint32_t column = 0; column < Dimensions; ++column)
    {
        const uint32_t next = (column + 1) % Dimensions, last = (column + Dimensions - 1) % Dimensions;
        for (bool positive : {true, false})
        {
            const double pair = double(matrix[column]) * matrix[AffineStride + (positive ? next : last)];
            const double final = matrix[AffineStride * 2 + (positive ? last : next)];
            const double product = pair * final, residual = std::fma(pair, final, -product);
            add(positive ? residual : -residual);
            add(positive ? product : -product);
        }
    }
    return length == 0;
}

bool Inverse(const GeometryInstance &instance, Matrix &inverse)
{
    const auto &matrix = instance.ObjectToWorld;
    for (float value : matrix)
        if (!std::isfinite(value))
            return false;
    if (Singular(matrix))
        return false;
    double augmented[Dimensions][AugmentedColumns]{};
    for (uint32_t row = 0; row < Dimensions; ++row)
    {
        double scale = 0;
        for (uint32_t column = 0; column < Dimensions; ++column)
            scale = std::max(scale, std::abs(double(matrix[row * AffineStride + column])));
        if (scale == 0)
            return false;
        for (uint32_t column = 0; column < Dimensions; ++column)
            augmented[row][column] = matrix[row * AffineStride + column] / scale;
        augmented[row][Dimensions + row] = 1 / scale;
    }
    for (uint32_t column = 0; column < Dimensions; ++column)
    {
        uint32_t pivot = column;
        for (uint32_t row = column + 1; row < Dimensions; ++row)
            if (std::abs(augmented[row][column]) > std::abs(augmented[pivot][column]))
                pivot = row;
        const double divisor = augmented[pivot][column];
        if (divisor == 0 || !std::isfinite(divisor))
            return false;
        for (uint32_t entry = 0; entry < AugmentedColumns; ++entry)
            std::swap(augmented[pivot][entry], augmented[column][entry]);
        for (uint32_t entry = 0; entry < AugmentedColumns; ++entry)
            augmented[column][entry] /= divisor;
        augmented[column][column] = 1;
        for (uint32_t row = 0; row < Dimensions; ++row)
        {
            if (row == column)
                continue;
            const double multiple = augmented[row][column];
            for (uint32_t entry = 0; entry < AugmentedColumns; ++entry)
                augmented[row][entry] = std::fma(-multiple, augmented[column][entry], augmented[row][entry]);
            augmented[row][column] = 0;
        }
    }
    for (uint32_t row = 0; row < Dimensions; ++row)
    {
        double translation = 0;
        for (uint32_t column = 0; column < Dimensions; ++column)
        {
            const double value = augmented[row][Dimensions + column];
            inverse[row * AffineStride + column] = value;
            translation = std::fma(-value, double(matrix[column * AffineStride + Dimensions]), translation);
        }
        inverse[row * AffineStride + Dimensions] = translation;
    }
    for (double value : inverse)
        if (!std::isfinite(value) || std::abs(value) > FloatLimit)
            return false;
    return true;
}

Coefficient Encode(double value)
{
    Coefficient result;
    int exponent = 0;
    const double significand = std::frexp(value, &exponent);
    result.High = float(significand);
    result.Low = float(significand - double(result.High));
    result.Exponent = exponent;
    return result;
}

bool WorldBounds(const Bounds &object, const GeometryInstance &instance, Record &record)
{
    Bounds world;
    Point roundoff{};
    for (uint32_t corner = 0; corner < CornerCount; ++corner)
    {
        Point point{}, transformed{};
        for (uint32_t axis = 0; axis < Dimensions; ++axis)
            point[axis] = (corner & (1u << axis)) ? object.High[axis] : object.Low[axis];
        for (uint32_t row = 0; row < Dimensions; ++row)
        {
            const auto &matrix = instance.ObjectToWorld;
            double value = matrix[row * AffineStride + Dimensions], magnitude = std::abs(value);
            for (uint32_t column = 0; column < Dimensions; ++column)
            {
                const double coefficient = matrix[row * AffineStride + column];
                value = std::fma(coefficient, point[column], value);
                magnitude += std::abs(coefficient * point[column]);
            }
            transformed[row] = value;
            roundoff[row] = std::max(roundoff[row], magnitude * BoundRoundoff);
        }
        world.Include(transformed);
    }
    for (uint32_t axis = 0; axis < Dimensions; ++axis)
    {
        const double low = world.Low[axis] - roundoff[axis], high = world.High[axis] + roundoff[axis];
        if (!std::isfinite(low) || !std::isfinite(high) || low < -FloatLimit || high > FloatLimit)
            return false;
        // One outward float step covers conversion rounding, including denormals.
        record.Minimum[axis] = std::nextafter(float(low), -std::numeric_limits<float>::infinity());
        record.Maximum[axis] = std::nextafter(float(high), std::numeric_limits<float>::infinity());
        if (!std::isfinite(record.Minimum[axis]) || !std::isfinite(record.Maximum[axis]))
            return false;
    }
    return true;
}

class Builder
{
    const Scene &SceneData;
    const std::function<bool()> &Cancelled;
    Hierarchy &Output;
    std::string &Error;
    std::unordered_map<const SceneGeometry *, ObjectInfo> Objects;

    bool Stop()
    {
        if (!Cancelled || !Cancelled())
            return false;
        Error = "Metal instance construction was cancelled.";
        return true;
    }
    bool Fail(const char *message)
    {
        Error = message;
        return false;
    }
    bool BoundObject(const SceneGeometry &geometry, Bounds &box)
    {
        const auto include = [&](Float3 point) {
            if (!std::isfinite(point.X) || !std::isfinite(point.Y) || !std::isfinite(point.Z))
                return false;
            box.Include({point.X, point.Y, point.Z});
            return true;
        };
        size_t visited = 0;
        for (const auto &triangle : geometry.Triangles)
        {
            if (visited++ % CancellationStride == 0 && Stop())
                return false;
            if (triangle.MaterialIndex >= SceneData.Materials.size() || !include(triangle.Position0) ||
                !include(triangle.Position1) || !include(triangle.Position2))
                return Fail("Metal instance triangle has an invalid position or material index.");
        }
        for (const auto &sphere : geometry.Spheres)
        {
            if (visited++ % CancellationStride == 0 && Stop())
                return false;
            if (sphere.MaterialIndex >= SceneData.Materials.size() || !std::isfinite(sphere.Radius) ||
                sphere.Radius <= 0 || !include(sphere.Center))
                return Fail("Metal instance sphere has an invalid center, radius or material index.");
            Point low{sphere.Center.X, sphere.Center.Y, sphere.Center.Z}, high = low;
            for (uint32_t axis = 0; axis < Dimensions; ++axis)
            {
                low[axis] -= sphere.Radius;
                high[axis] += sphere.Radius;
            }
            box.Include(low);
            box.Include(high);
        }
        for (uint32_t axis = 0; axis < Dimensions; ++axis)
            if (box.Low[axis] < -FloatLimit || box.High[axis] > FloatLimit)
                return Fail("Metal instance object bounds exceed the native float range.");
        return true;
    }
    bool Append(const SceneGeometry &geometry, const GeometryInstance &instance, uint32_t firstPrimitive)
    {
        if (Stop())
            return false;
        Matrix inverse{};
        if (!Inverse(instance, inverse))
            return Fail(
                "Metal instance transform is singular, non-finite or has an unrepresentable inverse.");
        if (PrimitiveCount(geometry) == 0)
            return true;
        auto found = Objects.find(&geometry);
        if (found == Objects.end())
        {
            ObjectInfo object;
            if (!BoundObject(geometry, object.Box))
                return false;
            object.Index = uint32_t(Output.Objects.size());
            found = Objects.emplace(&geometry, object).first;
            Output.Objects.push_back(&geometry);
        }
        Record record;
        record.ObjectIndex = found->second.Index;
        record.Motion = instance.Motion;
        record.Geometry.FirstPrimitive = firstPrimitive;
        record.Geometry.TriangleCount = uint32_t(geometry.Triangles.size());
        for (size_t entry = 0; entry < inverse.size(); ++entry)
            record.Inverse[entry] = Encode(inverse[entry]);
        const auto& rest = found->second.Box;
        Bounds swept;
        if (!BoundMotion(instance.Motion, rest.Low, rest.High, swept.Low, swept.High))
            return Fail("Metal motion has invalid swept bounds.");
        if (!WorldBounds(swept, instance, record))
            return Fail("Metal instance world bounds exceed the native float range.");
        Output.Records.push_back(record);
        return true;
    }
    static double Center(const Record &record, uint32_t axis)
    {
        return (double(record.Minimum[axis]) + record.Maximum[axis]) / 2;
    }
    bool Partition(uint32_t begin, uint32_t end, uint32_t depth)
    {
        if (Stop())
            return false;
        const uint32_t index = uint32_t(Output.Nodes.size()), count = end - begin;
        Bounds bounds;
        Point mean{}, variance{};
        for (uint32_t entry = begin; entry < end; ++entry)
        {
            if ((entry - begin) % CancellationStride == 0 && Stop())
                return false;
            const Record &record = Output.Records[entry];
            bounds.Include({record.Minimum[0], record.Minimum[1], record.Minimum[2]});
            bounds.Include({record.Maximum[0], record.Maximum[1], record.Maximum[2]});
            for (uint32_t axis = 0; axis < Dimensions; ++axis)
            {
                const double center = Center(record, axis), delta = center - mean[axis];
                mean[axis] += delta / (entry - begin + 1);
                variance[axis] += delta * (center - mean[axis]);
            }
        }
        Node node;
        for (uint32_t axis = 0; axis < Dimensions; ++axis)
        {
            node.Minimum[axis] = float(bounds.Low[axis]);
            node.Maximum[axis] = float(bounds.High[axis]);
        }
        Output.Nodes.push_back(node);
        if (count <= LeafCapacity || depth >= MaximumDepth)
        {
            Output.Nodes[index].FirstOrEscape = begin;
            Output.Nodes[index].Count = count;
            return true;
        }
        const uint32_t axis = uint32_t(std::max_element(variance.begin(), variance.end()) - variance.begin());
        const uint32_t middle = begin + count / 2;
        std::nth_element(Output.Records.begin() + begin, Output.Records.begin() + middle,
                         Output.Records.begin() + end, [axis](const Record &left, const Record &right) {
                             const double a = Center(left, axis), b = Center(right, axis);
                             return a != b ? a < b
                                           : left.Geometry.FirstPrimitive < right.Geometry.FirstPrimitive;
                         });
        if (!Partition(begin, middle, depth + 1) || !Partition(middle, end, depth + 1))
            return false;
        Output.Nodes[index].FirstOrEscape = uint32_t(Output.Nodes.size());
        return true;
    }

  public:
    Builder(const Scene &scene, const std::function<bool()> &cancelled, Hierarchy &output, std::string &error)
        : SceneData(scene), Cancelled(cancelled), Output(output), Error(error)
    {
    }
    bool Run()
    {
        if (Stop())
            return false;
        if (SceneData.Instances.size() >= MaximumRecords ||
            SceneData.Geometries.size() > std::numeric_limits<uint32_t>::max() ||
            PrimitiveCount(SceneData) > LocalPrimitiveLimit)
            return Fail("Metal instance counts exceed the native index layout.");
        for (size_t index = 0; index < SceneData.Geometries.size(); ++index)
        {
            if (index % CancellationStride == 0 && Stop())
                return false;
            const auto &geometry = SceneData.Geometries[index];
            if (!geometry || PrimitiveCount(*geometry) > LocalPrimitiveLimit)
                return Fail("Metal instance geometry is null or exceeds local indexing.");
        }
        uint64_t primitiveCount = PrimitiveCount(SceneData);
        if (!Append(SceneData, GeometryInstance{}, 0))
            return false;
        for (const auto &instance : SceneData.Instances)
        {
            if (instance.GeometryIndex >= SceneData.Geometries.size())
                return Fail("Metal instance geometry index is invalid.");
            const SceneGeometry &geometry = *SceneData.Geometries[instance.GeometryIndex];
            const uint64_t nextCount = primitiveCount + PrimitiveCount(geometry);
            if (nextCount > InvalidRayPrimitiveIndex)
                return Fail("Metal instance primitive IDs exceed 32 bits.");
            if (!Append(geometry, instance, uint32_t(primitiveCount)))
                return false;
            primitiveCount = nextCount;
        }
        if (!Output.Records.empty() && !Partition(0, uint32_t(Output.Records.size()), 0))
            return false;
        return !Stop();
    }
};
} // namespace

bool Validate(const Scene &scene, std::string &error)
{
    error.clear();
    if (scene.Instances.size() >= MaximumRecords ||
        scene.Geometries.size() > std::numeric_limits<uint32_t>::max() ||
        PrimitiveCount(scene) > LocalPrimitiveLimit)
    {
        error = "Metal instance counts exceed the native index layout.";
        return false;
    }
    for (const auto &geometry : scene.Geometries)
        if (!geometry || PrimitiveCount(*geometry) > LocalPrimitiveLimit)
        {
            error = "Metal instance geometry is null or exceeds local indexing.";
            return false;
        }
    uint64_t count = PrimitiveCount(scene);
    for (const auto &instance : scene.Instances)
    {
        if (!ValidateRigidMotion(instance.Motion, error))
            return false;
        if (instance.GeometryIndex >= scene.Geometries.size())
        {
            error = "Metal instance geometry index is invalid.";
            return false;
        }
        count += PrimitiveCount(*scene.Geometries[instance.GeometryIndex]);
        Matrix inverse{};
        if (count > InvalidRayPrimitiveIndex || !Inverse(instance, inverse))
        {
            error = "Metal instance has invalid primitive indexing or an unrepresentable affine inverse.";
            return false;
        }
    }
    return true;
}

bool Build(const Scene &scene, const std::function<bool()> &cancelled, Hierarchy &output, std::string &error)
{
    output = {};
    error.clear();
    Hierarchy pending;
    if (!Builder(scene, cancelled, pending, error).Run())
        return false;
    output = std::move(pending);
    return true;
}
} // namespace Arvo::RayTracing::MetalInstances
