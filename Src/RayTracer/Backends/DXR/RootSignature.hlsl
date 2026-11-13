// Existing surface/query bindings keep their indices. BDPT appends source,
// subpath, splat and error buffers plus eight batch constants. A separate medium
// CBV appends two DWORDs without changing the existing frame or BDPT controls.
// Path tracing appends one per-lane clock UAV; the root consumes 62 DWORDs.
// The host consumes this compiled blob; shader verification sees the same ABI.
#define ArvoDxrRootSignature \
    "SRV(t0),UAV(u0),UAV(u1),SRV(t1),SRV(t2),SRV(t3),SRV(t4),SRV(t5),SRV(t6),CBV(b0)," \
    "UAV(u2),SRV(t7),SRV(t8),UAV(u3),SRV(t9),SRV(t10),SRV(t11),SRV(t12),UAV(u4)," \
    "SRV(t13),UAV(u5),UAV(u6),UAV(u7),RootConstants(num32BitConstants=8,b1),CBV(b2),SRV(t14),SRV(t15),UAV(u8)"
