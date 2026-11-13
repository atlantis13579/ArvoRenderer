// Root descriptors occupy 56 DWORDs; b1 holds the bounded-work controls.
#define ArvoDxrPointGiRootSignature \
    "SRV(t0),UAV(u0),UAV(u1),SRV(t1),SRV(t2),SRV(t3),SRV(t4),SRV(t5),SRV(t6),CBV(b0)," \
    "UAV(u2),SRV(t7),SRV(t8),UAV(u3),SRV(t9),SRV(t10),SRV(t11),SRV(t12),UAV(u4)," \
    "SRV(t13),SRV(t14),SRV(t15),SRV(t16),UAV(u5),UAV(u6),UAV(u7),UAV(u8),CBV(b1)"
