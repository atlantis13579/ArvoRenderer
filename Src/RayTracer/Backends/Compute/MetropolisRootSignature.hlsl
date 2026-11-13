// MLT has a dedicated chain/bootstrap table; the frame stays in an upload CBV.
#define ArvoComputeMetropolisRootSignature \
    "DescriptorTable(SRV(t0,numDescriptors=8,offset=0),SRV(t9,numDescriptors=2,offset=8))," \
    "DescriptorTable(UAV(u0,numDescriptors=12,offset=0)),CBV(b0),RootConstants(num32BitConstants=12,b1)"
