// AABB traversal remains hardware accelerated. Only moving triangles use this
// procedural intersection; static meshes retain fixed-function intersection.
// ReportHit carries the same barycentric attributes as the static triangle hit.
[shader("intersection")]
void MotionTriangleIntersection()
{
    const InstanceRecord instance = InstanceRecords[InstanceID()];
    double3 origin = double3(ObjectRayOrigin()), direction = double3(ObjectRayDirection());
    const MotionPose pose = InterpolateMotion(instance.Motion.Transform, InstanceSceneTime());
    InverseInstanceMotion(pose, origin, direction);
    float distance;
    BuiltInTriangleIntersectionAttributes attributes;
    if (IntersectTriangle(instance.Geometry.TriangleOffset + PrimitiveIndex(), float3(origin),
        float3(direction), RayTMin(), RayTCurrent(), distance, attributes.barycentrics))
        ReportHit(distance, 0, attributes);
}
