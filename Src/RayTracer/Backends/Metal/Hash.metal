// Metal's counter-stream avalanche. Sample domains supply the seed; no mutable
// generator state depends on command submission order or thread scheduling.
uint Hash(uint value)
{
    value ^= value >> 16u;
    value *= AvalancheMultiplierA;
    value ^= value >> 15u;
    value *= AvalancheMultiplierB;
    value ^= value >> 16u;
    return value;
}
float Random01(uint seed)
{
    return float(Hash(seed) & RandomMantissaMask) / RandomMantissaRange;
}
