#ifndef ARVO_DXR_BSDF_TYPES
#define ARVO_DXR_BSDF_TYPES
// Data contracts: local unit directions, view.z > 0; delta PDFs are masses.
static const uint ScatteringReflection = 1u, ScatteringTransmission = 2u;
static const uint ScatteringDiffuse = 4u, ScatteringGlossy = 8u, ScatteringDelta = 16u;
static const uint TransportRadiance = 0u, TransportImportance = 1u;
struct BsdfEvaluation
{
    float Value, Pdf, ReversePdf;
};
struct BsdfSample
{
    float3 Direction;
    float Value, Pdf, ReversePdf, RelativeIndex;
    uint Event;
};
BsdfEvaluation BsdfValue(float value, float pdf, float reversePdf)
{
    BsdfEvaluation result;
    result.Value = value;
    result.Pdf = pdf;
    result.ReversePdf = reversePdf;
    return result;
}
BsdfSample BsdfEvent(float3 direction, BsdfEvaluation evaluation, float relativeIndex, uint event)
{
    BsdfSample result;
    result.Direction = direction;
    result.Value = evaluation.Value;
    result.Pdf = evaluation.Pdf;
    result.ReversePdf = evaluation.ReversePdf;
    result.RelativeIndex = relativeIndex;
    result.Event = event;
    return result;
}
#endif
