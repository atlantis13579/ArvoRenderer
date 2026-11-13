// Dedicated BDPT bindings. Frame data uses a CBV rather than inline root DWORDs,
// leaving room for the full-image path batch and emission-domain constants.
#define ArvoComputeBidirectionalRootSignature \
    "DescriptorTable(SRV(t0,numDescriptors=8,offset=0),SRV(t9,numDescriptors=2,offset=8))," \
    "DescriptorTable(UAV(u0,numDescriptors=8,offset=0)),CBV(b0),RootConstants(num32BitConstants=8,b1)"
