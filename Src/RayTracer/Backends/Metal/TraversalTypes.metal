#include "MotionData.h"
// Data-only traversal views. BLAS child/reference indices stay local;
// FirstPrimitive identifies the placement in the public hit ABI.
struct EncodedRay
{
    uint4 OriginMinimum;
    uint4 DirectionMaximum;
};
struct InstanceGeometryRange
{
    uint NodeBegin, NodeCount, ReferenceBegin, TriangleBegin;
    uint SphereBegin, TriangleCount, FirstPrimitive, Reserved;
};
struct InstanceRecord
{
    WideMath::Number Inverse[12];
    InstanceGeometryRange Geometry;
    packed_uint3 Minimum;
    uint ObjectIndex;
    packed_uint3 Maximum;
    uint Reserved;
    MotionRigid Motion;
};
struct InstanceNode
{
    packed_uint3 Minimum;
    uint FirstOrEscape;
    packed_uint3 Maximum;
    uint Count;
};
struct ObjectGeometry
{
    device const PackedAccelerationNode *Nodes;
    device const uint *References;
    device const PackedTriangle *Triangles;
    device const PackedSphere *Spheres;
    uint NodeCount, TriangleCount, Acceleration;
    // Per-path observation time, independent of the affine ray parameter.
    float SceneTime;
};
struct SceneGeometryBuffers
{
    ObjectGeometry Object;
    device const InstanceRecord *Instances;
    device const InstanceNode *Nodes;
    device atomic_uint *Error;
    uint NodeCount;
};
static_assert(sizeof(InstanceRecord) == 336 && sizeof(InstanceNode) == 32,
              "Metal instance buffer ABI mismatch");
