// Keep scene indices compatible with other DXR pipelines. One raw arena holds
// chains, bootstrap samples and the CDF: 46 descriptor + 16 constant DWORDs.
#define ArvoDxrMetropolisRootSignature \
    "SRV(t0),UAV(u0),UAV(u1),SRV(t1),SRV(t2),SRV(t3),SRV(t4),SRV(t5),SRV(t6),CBV(b0)," \
    "UAV(u2),SRV(t7),SRV(t8),UAV(u3),SRV(t9),SRV(t10),SRV(t11),SRV(t12),UAV(u4)," \
    "SRV(t13),UAV(u5),UAV(u6),UAV(u7),RootConstants(num32BitConstants=16,b1)"
