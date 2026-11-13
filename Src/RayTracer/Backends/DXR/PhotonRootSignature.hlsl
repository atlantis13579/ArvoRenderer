// Preserve all scene roots. 23 descriptors + 16 constants use 62 of 64 DWORDs.
#define ArvoDxrPhotonRootSignature \
    "SRV(t0),UAV(u0),UAV(u1),SRV(t1),SRV(t2),SRV(t3),SRV(t4),SRV(t5),SRV(t6),CBV(b0)," \
    "UAV(u2),SRV(t7),SRV(t8),UAV(u3),SRV(t9),SRV(t10),SRV(t11),SRV(t12),UAV(u4)," \
    "SRV(t13),UAV(u5),UAV(u6),UAV(u7),RootConstants(num32BitConstants=16,b1)"
