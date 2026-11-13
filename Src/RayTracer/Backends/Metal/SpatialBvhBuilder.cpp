#include "SpatialBvhBuilder.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Arvo::RayTracing::MetalSpatial
{
namespace
{
// Metal's SBVH is built independently with bucketed object SAH and uniform
// spatial candidates. Only references are cut: the original surface remains
// intact. Conservative box clipping also supports analytic spheres. Bvh.metal
// consumes the resulting depth-first escape links without a traversal stack.
// Stich et al., "Spatial Splits in Bounding Volume Hierarchies" (2009), Section 4.
// https://www.nvidia.com/docs/IO/77714/sbvh.pdf
constexpr uint32_t BinCount = 16, PreferredLeafSize = 4, DepthLimit = 48;
constexpr uint64_t ReferenceExpansionLimit = 8;
constexpr size_t CancellationInterval = 1024;
constexpr double TraversalWeight = 1.0;
struct Box
{
    std::array<float, 3> Low, High;
    Box()
    {
        Low.fill(std::numeric_limits<float>::max());
        High.fill(std::numeric_limits<float>::lowest());
    }
    void Add(const Reference &reference)
    {
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            Low[axis] = std::min(Low[axis], reference.Lower[axis]);
            High[axis] = std::max(High[axis], reference.Upper[axis]);
        }
    }
    void Add(const Box &box)
    {
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            Low[axis] = std::min(Low[axis], box.Low[axis]);
            High[axis] = std::max(High[axis], box.High[axis]);
        }
    }
    double Area() const
    {
        const double x = double(High[0]) - Low[0], y = double(High[1]) - Low[1], z = double(High[2]) - Low[2];
        return 2 * (x * y + y * z + z * x);
    }
};
double Midpoint(const Reference &reference, uint32_t axis)
{
    return (double(reference.Lower[axis]) + reference.Upper[axis]) / 2;
}
uint32_t Bucket(double center, double first, double span)
{
    return std::min(BinCount - 1, uint32_t(std::max(0.0, (center - first) * BinCount / span)));
}
struct Split
{
    bool Valid = false, Spatial = false;
    uint32_t Axis = 0, LastBucket = 0;
    double Origin = 0, Extent = 0, Cost = 0;
    float Plane = 0;
};
class Builder
{
    Hierarchy &Result;
    const std::function<bool()> &Cancelled;
    std::string &Error;
    uint64_t ReservedReferences = 0, ReferenceLimit = 0, ReservedNodes = 1;
    size_t Work = 0;
    bool Stop()
    {
        if (Cancelled && Cancelled())
        {
            Error = "Metal spatial BVH build was cancelled";
            return true;
        }
        return false;
    }
    bool Poll() { return ++Work % CancellationInterval == 0 && Stop(); }
    bool Emit(std::vector<Reference> pieces, uint32_t depth)
    {
        if (Stop())
            return false;
        Box bounds;
        for (const Reference &piece : pieces)
        {
            if (Poll())
                return false;
            bounds.Add(piece);
        }
        const uint32_t nodeIndex = uint32_t(Result.Nodes.size());
        Result.Nodes.push_back({bounds.Low, bounds.High, 0, 0});
        Split best;
        const double parentArea = bounds.Area();
        best.Cost = parentArea * pieces.size();
        if (pieces.size() > PreferredLeafSize && depth < DepthLimit && parentArea > 0 &&
            ReservedNodes + 2 < UINT32_MAX)
        {
            for (uint32_t axis = 0; axis < 3; ++axis)
            {
                double minimum = std::numeric_limits<double>::max(), maximum = -minimum;
                for (const Reference &piece : pieces)
                {
                    if (Poll())
                        return false;
                    const double center = Midpoint(piece, axis);
                    minimum = std::min(minimum, center);
                    maximum = std::max(maximum, center);
                }
                const double span = maximum - minimum;
                if (span > 0)
                {
                    struct Bin
                    {
                        Box Bounds;
                        uint64_t Count = 0;
                    };
                    std::array<Bin, BinCount> bins;
                    for (const Reference &piece : pieces)
                    {
                        if (Poll())
                            return false;
                        Bin &bin = bins[Bucket(Midpoint(piece, axis), minimum, span)];
                        bin.Bounds.Add(piece);
                        ++bin.Count;
                    }
                    // There are few buckets; a direct prefix/suffix sweep keeps
                    // the candidate bookkeeping visible for this teaching backend.
                    for (uint32_t cut = 0; cut + 1 < BinCount; ++cut)
                    {
                        Box left, right;
                        uint64_t leftCount = 0, rightCount = 0;
                        for (uint32_t bin = 0; bin < BinCount; ++bin)
                        {
                            if (bins[bin].Count == 0)
                                continue;
                            if (bin <= cut)
                            {
                                left.Add(bins[bin].Bounds);
                                leftCount += bins[bin].Count;
                            }
                            else
                            {
                                right.Add(bins[bin].Bounds);
                                rightCount += bins[bin].Count;
                            }
                        }
                        if (leftCount == 0 || rightCount == 0)
                            continue;
                        const double cost = TraversalWeight * parentArea + left.Area() * leftCount +
                                            right.Area() * rightCount;
                        if (cost < best.Cost)
                            best = {true, false, axis, cut, minimum, span, cost, 0};
                    }
                }
                const double width = double(bounds.High[axis]) - bounds.Low[axis];
                for (uint32_t cut = 1; cut < BinCount && width > 0; ++cut)
                {
                    const float plane = float(bounds.Low[axis] + width * cut / BinCount);
                    if (!(plane > bounds.Low[axis] && plane < bounds.High[axis]))
                        continue;
                    Box left, right;
                    uint64_t leftCount = 0, rightCount = 0;
                    for (const Reference &piece : pieces)
                    {
                        if (Poll())
                            return false;
                        if (piece.Upper[axis] <= plane)
                        {
                            left.Add(piece);
                            ++leftCount;
                        }
                        else if (piece.Lower[axis] >= plane)
                        {
                            right.Add(piece);
                            ++rightCount;
                        }
                        else
                        {
                            Reference a = piece, b = piece;
                            a.Upper[axis] = plane;
                            b.Lower[axis] = plane;
                            left.Add(a);
                            right.Add(b);
                            ++leftCount;
                            ++rightCount;
                        }
                    }
                    if (!leftCount || !rightCount ||
                        leftCount + rightCount - pieces.size() > ReferenceLimit - ReservedReferences)
                        continue;
                    const double cost =
                        TraversalWeight * parentArea + left.Area() * leftCount + right.Area() * rightCount;
                    if (cost < best.Cost)
                        best = {true, true, axis, 0, 0, 0, cost, plane};
                }
            }
        }
        if (!best.Valid)
        {
            Result.Nodes[nodeIndex].FirstOrEscape = uint32_t(Result.References.size());
            Result.Nodes[nodeIndex].Count = uint32_t(pieces.size());
            for (const Reference &piece : pieces)
            {
                if (Poll())
                    return false;
                Result.References.push_back(piece.EncodedIndex);
            }
            return true;
        }
        std::vector<Reference> left, right;
        for (const Reference &piece : pieces)
        {
            if (Poll())
                return false;
            if (!best.Spatial)
            {
                (Bucket(Midpoint(piece, best.Axis), best.Origin, best.Extent) <= best.LastBucket ? left
                                                                                                 : right)
                    .push_back(piece);
            }
            else if (piece.Upper[best.Axis] <= best.Plane)
                left.push_back(piece);
            else if (piece.Lower[best.Axis] >= best.Plane)
                right.push_back(piece);
            else
            {
                Reference a = piece, b = piece;
                a.Upper[best.Axis] = best.Plane;
                b.Lower[best.Axis] = best.Plane;
                left.push_back(a);
                right.push_back(b);
            }
        }
        if (best.Spatial)
            ++Result.SpatialSplits;
        ReservedReferences += left.size() + right.size() - pieces.size();
        ReservedNodes += 2;
        if (!Emit(std::move(left), depth + 1) || !Emit(std::move(right), depth + 1))
            return false;
        Result.Nodes[nodeIndex].FirstOrEscape = uint32_t(Result.Nodes.size());
        return true;
    }

  public:
    Builder(Hierarchy &output, const std::function<bool()> &cancellation, std::string &error)
        : Result(output), Cancelled(cancellation), Error(error)
    {
    }
    bool Run(const std::vector<Reference> &input)
    {
        if (Stop())
            return false;
        if (input.size() >= UINT32_MAX)
        {
            Error = "Metal spatial BVH reference limit exceeded";
            return false;
        }
        for (const Reference &piece : input)
        {
            if (Poll())
                return false;
            for (uint32_t axis = 0; axis < 3; ++axis)
                if (!std::isfinite(piece.Lower[axis]) || !std::isfinite(piece.Upper[axis]) ||
                    piece.Lower[axis] > piece.Upper[axis])
                {
                    Error = "Invalid Metal spatial BVH bounds";
                    return false;
                }
        }
        ReservedReferences = input.size();
        ReferenceLimit = std::min<uint64_t>(UINT32_MAX, ReservedReferences * ReferenceExpansionLimit);
        return input.empty() || (Emit(input, 0) && !Stop());
    }
};
} // namespace
bool Build(const std::vector<Reference> &references, const std::function<bool()> &cancelled,
           Hierarchy &hierarchy, std::string &error)
{
    hierarchy = {};
    if (Builder(hierarchy, cancelled, error).Run(references))
        return true;
    hierarchy = {};
    return false;
}
} // namespace Arvo::RayTracing::MetalSpatial
