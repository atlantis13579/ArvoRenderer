// Preparation is a ray-generation dispatch without a TLAS or hit groups.
// Root addresses select each bounded palette batch; coefficients retain status.
#define ArvoDxrSpectralRootSignature "SRV(t0),SRV(t1),SRV(t2),SRV(t3),SRV(t4),UAV(u0),UAV(u1),UAV(u2)"
