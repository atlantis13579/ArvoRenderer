// Each RayGen invocation counts locally, then adds one uint2 delta per statistic.
// Global low-word addition returns the previous value so exactly one carry is
// added when that addition wraps. Consumers read after the dispatch fence; no
// native 64-bit atomics or newer shader model is required.
static const uint PrimaryCounterWord = 0u;
static const uint ShadowCounterWord = 2u;
static const uint ReflectionCounterWord = 4u;
static const uint CompletedCounterWord = 6u;
static const uint CounterCarryThreshold = 0xffffffffu;

void IncrementLocalCounter(inout uint2 counter)
{
    ++counter.x;
    counter.y += counter.x == 0u ? 1u : 0u;
}

void AddStatistic(uint lowWord, uint2 value)
{
    uint carry = 0u;
    if (value.x != 0u)
    {
        uint previous;
        InterlockedAdd(Statistics[lowWord], value.x, previous);
        carry = previous > CounterCarryThreshold - value.x ? 1u : 0u;
    }
    const uint highIncrement = value.y + carry;
    if (highIncrement != 0u)
        InterlockedAdd(Statistics[lowWord + 1u], highIncrement);
}
