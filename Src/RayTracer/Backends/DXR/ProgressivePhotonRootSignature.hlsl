// 26 root descriptors use 52 DWORDs. A CBV keeps persistent-state bindings below
// the 64-DWORD limit; controls are updated only after the previous dispatch retires.
// https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signature-limits
#define ArvoDxrPpmRootSignature \
    "SRV(t0),UAV(u0),UAV(u1),SRV(t1),SRV(t2),SRV(t3),SRV(t4),SRV(t5),SRV(t6),CBV(b0)," \
    "UAV(u2),SRV(t7),SRV(t8),UAV(u3),SRV(t9),SRV(t10),SRV(t11),SRV(t12),UAV(u4)," \
    "SRV(t13),UAV(u5),UAV(u6),UAV(u7),UAV(u8),UAV(u9),CBV(b1)"
