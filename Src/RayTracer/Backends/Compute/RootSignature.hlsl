// Production binding data, compiled and verified against every Compute kernel.
// Tables cost 2 DWORDs, root buffers 4, and the frame/medium CBVs 4.
// Frame constants outgrew the root-constant budget when observation seeds were added.
// Root parameter indices match BindCommon/BindDispatch in the Compute RHI.
#define ArvoComputeRootSignature \
    "DescriptorTable(SRV(t0,numDescriptors=8,offset=0),SRV(t9,numDescriptors=4,offset=8))," \
    "DescriptorTable(UAV(u0,numDescriptors=5,offset=0),UAV(u6,numDescriptors=1,offset=5))," \
    "CBV(b0),SRV(t8),UAV(u5),CBV(b1)"
