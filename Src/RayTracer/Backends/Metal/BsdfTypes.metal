// Scalar local-frame BSDF contracts. Delta PDFs are discrete masses, not sr^-1.
constant uint ScatteringReflection = 1u, ScatteringTransmission = 2u;
constant uint ScatteringDiffuse = 4u, ScatteringGlossy = 8u, ScatteringDelta = 16u;
constant uint TransportRadiance = 0u, TransportImportance = 1u;
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
