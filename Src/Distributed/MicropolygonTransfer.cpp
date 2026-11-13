#include "Distributed/MicropolygonTransfer.h"
#include "Distributed/Protocol.h"
#include "Renderer/MicropolygonData.h"
#include <charconv>

namespace Arvo::Distributed
{
namespace
{
namespace M = RayTracing::MicropolygonData;
constexpr char PlanAlphabet[] = "0123456789abcdefghijklmnopqrstuv";
constexpr size_t FingerprintDigits = sizeof(uint64_t) * 2;
void ValidateWord(uint32_t word)
{
    if ((word & ~(M::LevelMask | M::LevelClamped)) || (word & M::LevelMask) > M::MaximumLevel)
        throw std::invalid_argument("Invalid micropolygon plan word");
}
} // namespace
std::string EncodeMicropolygons(const RayTracing::MicropolygonOptions& options)
{
    std::string error;
    if (!RayTracing::ValidateMicropolygonOptions(options, error))
        throw std::invalid_argument(error);
    return std::to_string(options.Enabled) + ',' + Protocol::Number(options.TargetEdgePixels) + ',' +
           std::to_string(options.MaximumLevel) + ',' + std::to_string(options.TriangleBudget) + ',' +
           std::to_string(options.DisplacementMaterial) + ',' + Protocol::Number(options.Amplitude) + ',' +
           Protocol::Number(options.Wavelength) + ',' + Protocol::Number(options.PhaseCycles);
}
RayTracing::MicropolygonOptions ParseMicropolygons(const std::string& text)
{
    constexpr size_t FieldCount = 8, MaximumOptionsBytes = 512;
    if (text.size() > MaximumOptionsBytes)
        throw std::invalid_argument("Oversized micropolygon options");
    std::map<std::string, std::string> fields;
    size_t begin = 0;
    for (size_t i = 0; i < FieldCount; ++i)
    {
        const size_t end = text.find(',', begin);
        if ((i + 1 < FieldCount) != (end != std::string::npos))
            throw std::invalid_argument("Expected eight micropolygon option fields");
        fields[std::to_string(i)] = text.substr(begin, end == std::string::npos ? end : end - begin);
        begin = end == std::string::npos ? text.size() : end + 1;
    }
    RayTracing::MicropolygonOptions result;
    result.Enabled = Protocol::UInt(fields, "0", 0, 1) != 0;
    result.TargetEdgePixels = Protocol::Float(fields, "1");
    result.MaximumLevel = Protocol::UInt(fields, "2", 0, M::MaximumLevel);
    result.TriangleBudget = Protocol::UInt(fields, "3", 1, M::MaximumOutputTriangles);
    result.DisplacementMaterial = Protocol::UInt(fields, "4", 0, std::numeric_limits<uint32_t>::max());
    result.Amplitude = Protocol::Float(fields, "5");
    result.Wavelength = Protocol::Float(fields, "6");
    result.PhaseCycles = Protocol::Float(fields, "7");
    std::string error;
    if (!RayTracing::ValidateMicropolygonOptions(result, error))
        throw std::invalid_argument(error);
    return result;
}
std::string EncodeDicingPlan(const RayTracing::MicropolygonPlan& plan)
{
    if (plan.Components.size() > M::MaximumOutputTriangles)
        throw std::length_error("Oversized micropolygon plan");
    // One URL-safe character per component: four level bits and one clamp bit.
    // Dense encoding bounds request size even for disconnected input triangles.
    std::string text(FingerprintDigits, '0');
    for (size_t digit = 0; digit < FingerprintDigits; ++digit)
        text[FingerprintDigits - 1 - digit] = PlanAlphabet[(plan.Fingerprint >> (digit * 4)) & 15u];
    text += ':';
    text.reserve(text.size() + plan.Components.size());
    for (uint32_t word : plan.Components)
    {
        ValidateWord(word);
        text += PlanAlphabet[(word & M::LevelMask) | ((word & M::LevelClamped) ? 16u : 0u)];
    }
    return text;
}
std::shared_ptr<const RayTracing::MicropolygonPlan> ParseDicingPlan(const std::string& text)
{
    if (text.size() < FingerprintDigits + 1 || text[FingerprintDigits] != ':' ||
        text.size() - FingerprintDigits - 1 > M::MaximumOutputTriangles)
        throw std::invalid_argument("Invalid micropolygon plan size");
    auto plan = std::make_shared<RayTracing::MicropolygonPlan>();
    const auto decoded = std::from_chars(text.data(), text.data() + FingerprintDigits, plan->Fingerprint, 16);
    if (decoded.ec != std::errc{} || decoded.ptr != text.data() + FingerprintDigits)
        throw std::invalid_argument("Invalid micropolygon fingerprint");
    plan->Components.reserve(text.size() - FingerprintDigits - 1);
    for (size_t i = FingerprintDigits + 1; i < text.size(); ++i)
    {
        const auto* code = std::strchr(PlanAlphabet, text[i]);
        if (!code || text[i] == '\0')
            throw std::invalid_argument("Invalid micropolygon level encoding");
        const auto value = uint32_t(code - PlanAlphabet);
        const uint32_t word = (value & 15u) | ((value & 16u) ? M::LevelClamped : 0);
        ValidateWord(word);
        plan->Components.push_back(word);
    }
    return plan;
}
void RequireMicropolygonWorker(const std::map<std::string, std::string>& fields)
{
    const auto version = fields.find("version"), capability = fields.find("micropolygon_wire");
    if (version == fields.end() || version->second != Protocol::Version || capability == fields.end() ||
        capability->second != MicropolygonWireVersion)
        throw std::runtime_error("Worker does not support native micropolygon planning");
}
} // namespace Arvo::Distributed
