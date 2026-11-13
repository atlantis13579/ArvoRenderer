// Keep the 19 common DXR scene roots in place. VCM appends its own source and
// path storage, controls and frame buffers: 32 roots, exactly 64 DWORDs.
#define ArvoDxrVcmRootSignature \
    "SRV(t0),UAV(u0),UAV(u1),SRV(t1),SRV(t2),SRV(t3),SRV(t4),SRV(t5),SRV(t6),CBV(b0)," \
    "UAV(u2),SRV(t7),SRV(t8),UAV(u3),SRV(t9),SRV(t10),SRV(t11),SRV(t12),UAV(u4)," \
    "SRV(t13),UAV(u5),UAV(u6),UAV(u7),UAV(u8),CBV(b1),SRV(t14),UAV(u9),CBV(b2)," \
    "UAV(u10),UAV(u11),UAV(u12),CBV(b3)"
