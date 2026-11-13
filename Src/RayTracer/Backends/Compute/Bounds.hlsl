// Slab intervals use the original ray parameter. A tiny nonzero direction is
// not parallel: it may reach a slab when the caller allows a sufficiently large t.
bool IntersectBounds(PackedBvhNode node, RayData ray, float closestDistance)
{
    float nearDistance = ray.MinimumDistance;
    float farDistance = min(ray.MaximumDistance, closestDistance);
    [unroll] for (uint axis = 0; axis < 3; ++axis)
    {
        const float origin = ray.Origin[axis];
        const float direction = ray.Direction[axis];
        if (direction == 0.0f)
        {
            if (origin < node.Minimum[axis] || origin > node.Maximum[axis]) return false;
            continue;
        }
        float distance0 = (node.Minimum[axis] - origin) / direction;
        float distance1 = (node.Maximum[axis] - origin) / direction;
        if (distance0 > distance1)
        {
            const float temporary = distance0;
            distance0 = distance1;
            distance1 = temporary;
        }
        nearDistance = max(nearDistance, distance0);
        farDistance = min(farDistance, distance1);
        if (nearDistance > farDistance) return false;
    }
    return true;
}
