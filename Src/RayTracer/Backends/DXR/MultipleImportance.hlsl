// DXR native MultipleImportance sampling. No other backend implementation is included.
// Veach (1997), ch. 9: PDFs include sample counts; the power heuristic squares their ratio.
float MisWeight(float sampled, float competing)
{
    const uint heuristic = (TransportFlags >> MisShift) & MisMask;
    if (heuristic == NoMis || competing <= 0.0f) return 1.0f;
    if (sampled <= 0.0f) return 0.0f;
    float ratio = competing / sampled;
    if (heuristic == PowerMis) ratio *= ratio;
    return 1.0f / (1.0f + ratio);
}
