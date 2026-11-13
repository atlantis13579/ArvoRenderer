// CUDA native MultipleImportance sampling. No other backend implementation is included.
// Veach (1997), ch. 9: PDFs include sample counts; the power heuristic squares their ratio.
    __device__ float MisWeight(float sampled, float competing, unsigned int flags)
    {
        const unsigned int mode = (flags >> MisShift) & MisMask;
        if (mode == NoMis || competing <= 0.0f) return 1.0f;
        if (sampled <= 0.0f) return 0.0f;
        float ratio = competing / sampled;
        if (mode == PowerMis) ratio *= ratio;
        return 1.0f / (1.0f + ratio);
    }
