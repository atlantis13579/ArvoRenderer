// Root addresses allow bounded palette batches without mutable descriptor
// heaps. Sources and coefficient outputs advance by their own element strides.
#define ArvoComputeSpectralRootSignature                                                                     \
    "SRV(t0),SRV(t1),SRV(t2),SRV(t3),SRV(t4),UAV(u0),UAV(u1),UAV(u2),"                                       \
    "RootConstants(num32BitConstants=2,b0)"
