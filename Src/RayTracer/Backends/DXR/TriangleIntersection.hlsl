// Moller and Trumbore (1997), "Fast, Minimum Storage Ray-Triangle Intersection".
// This independent DXR leaf serves software traversal and procedural motion.
// Static hardware BVH triangles retain fixed-function intersection.
bool IntersectTriangle(uint primitive, float3 origin, float3 direction,
    float minimumDistance, float maximumDistance, out float distance, out float2 barycentrics)
{
    distance = maximumDistance;
    barycentrics = 0.0;
    uint vertex = primitive * 3;
    float3 first = Vertices[vertex].xyz;
    float3 edgeU = Vertices[vertex + 1].xyz - first;
    float3 edgeV = Vertices[vertex + 2].xyz - first;
    float directionScale = max(abs(direction.x), max(abs(direction.y), abs(direction.z)));
    float3 conditionedDirection = direction / directionScale;
    float3 perpendicular = cross(conditionedDirection, edgeV);
    float determinant = dot(edgeU, perpendicular);
    // No fixed epsilon: rescaling a valid ray must not change its hits.
    if (determinant == 0.0 || !isfinite(determinant)) return false;
    float3 offset = origin - first;
    float inverseDeterminant = 1.0 / determinant;
    float u = dot(offset, perpendicular) * inverseDeterminant;
    if (u < 0.0 || u > 1.0) return false;
    float3 offsetCross = cross(offset, edgeU);
    float v = dot(conditionedDirection, offsetCross) * inverseDeterminant;
    if (v < 0.0 || u + v > 1.0) return false;
    distance = (dot(edgeV, offsetCross) * inverseDeterminant) / directionScale;
    if (!isfinite(distance) || distance <= minimumDistance || distance >= maximumDistance) return false;
    barycentrics = float2(u, v);
    return true;
}
