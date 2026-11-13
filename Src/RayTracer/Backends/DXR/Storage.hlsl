// Only the DXR software acceleration buffer contract is shared. Each algorithm
// owns its node layout and loader: kd nodes occupy 16 bytes; BIH and SBVH 32 bytes.
// Raw storage avoids binding overlapping structured buffers with different strides.
ByteAddressBuffer SoftwareNodes : register(t9);
StructuredBuffer<uint> SoftwarePrimitiveIndices : register(t10);
