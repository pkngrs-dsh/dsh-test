"""NumPy-only nonadjacent triangle intersection gate for client refinement.

AABB broad phase plus plane clipping; shared-index neighbors are excluded.
This is a numerical regression gate, not a collision-free certification.
Coordinates and reported lengths use the caller's canonical frame.
"""
from __future__ import annotations
import numpy as np

def broadPhase(vertices, faces, epsilon):
    triangles = vertices[faces]
    lower, upper = triangles.min(axis=1), triangles.max(axis=1)
    order = np.argsort(lower[:, 0], kind="stable")
    sortedX = lower[order, 0]
    pairs, removedNeighbors = [], 0
    for position, faceId in enumerate(order):
        end = np.searchsorted(sortedX, upper[faceId, 0] + epsilon, side="right")
        other = order[position + 1:end]
        if not len(other):
            continue
        overlap = ((lower[other, 1:] <= upper[faceId, 1:] + epsilon)
                   & (upper[other, 1:] >= lower[faceId, 1:] - epsilon)).all(axis=1)
        other = other[overlap]
        shared = (faces[faceId][None, :, None] == faces[other][:, None, :]).any(axis=(1, 2))
        removedNeighbors += int(shared.sum())
        other = other[~shared]
        pairs.extend((min(int(faceId), int(j)), max(int(faceId), int(j))) for j in other)
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2), removedNeighbors


def _planeSection(triangle, distances, epsilon):
    points = [triangle[i] for i in range(3) if abs(distances[i]) <= epsilon]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        if ((distances[i] > epsilon and distances[j] < -epsilon)
                or (distances[j] > epsilon and distances[i] < -epsilon)):
            weight = distances[i] / (distances[i] - distances[j])
            points.append(triangle[i] + weight * (triangle[j] - triangle[i]))
    return np.asarray(points).reshape(-1, 3)


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _coplanarOverlap(a, b, normal, epsilon):
    axis = int(np.argmax(np.abs(normal)))
    kept = [i for i in range(3) if i != axis]
    polygon = [p.copy() for p in a[:, kept]]
    clip = b[:, kept]
    orientation = 1. if _cross2(clip[1] - clip[0], clip[2] - clip[0]) >= 0 else -1.
    for first, second in ((0, 1), (1, 2), (2, 0)):
        if not polygon:
            return None
        edge = clip[second] - clip[first]
        tolerance = epsilon * max(float(np.linalg.norm(edge)), epsilon)
        output = []
        previous = polygon[-1]
        oldDistance = orientation * _cross2(edge, previous - clip[first])
        for current in polygon:
            distance = orientation * _cross2(edge, current - clip[first])
            oldInside, inside = oldDistance >= -tolerance, distance >= -tolerance
            if inside != oldInside:
                denominator = oldDistance - distance
                if abs(denominator) > 1e-30:
                    output.append(previous + oldDistance / denominator * (current - previous))
            if inside:
                output.append(current)
            previous, oldDistance = current, distance
        polygon = output
    if not polygon:
        return None
    polygon = np.asarray(polygon)
    area2 = abs(sum(_cross2(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon)))) / 2
    area = area2 / max(abs(float(normal[axis])), 1e-30)
    point = np.zeros(3)
    point[kept] = polygon.mean(axis=0)
    point[axis] = (np.dot(normal, a[0]) - np.dot(normal[kept], point[kept])) / normal[axis]
    return {"kind": "coplanarAreaOverlap" if area > epsilon * epsilon else "coplanarContact",
            "point": point.tolist(), "overlapArea": float(area), "bothPlanesStraddled": False}


def intersectTriangles(a, b, epsilon):
    na = np.cross(a[1] - a[0], a[2] - a[0])
    nb = np.cross(b[1] - b[0], b[2] - b[0])
    la, lb = np.linalg.norm(na), np.linalg.norm(nb)
    if la <= epsilon * epsilon or lb <= epsilon * epsilon:
        return None
    na, nb = na / la, nb / lb
    da, db = (a - b[0]) @ nb, (b - a[0]) @ na
    if any(np.min(values) > epsilon or np.max(values) < -epsilon for values in (da, db)):
        return None
    direction = np.cross(na, nb)
    length = np.linalg.norm(direction)
    if length <= 1e-10:
        if max(float(np.max(np.abs(da))), float(np.max(np.abs(db)))) > epsilon:
            return None
        return _coplanarOverlap(a, b, na, epsilon)
    direction /= length
    sa, sb = _planeSection(a, da, epsilon), _planeSection(b, db, epsilon)
    if not len(sa) or not len(sb):
        return None
    aa, bb = sa @ direction, sb @ direction
    low, high = max(float(aa.min()), float(bb.min())), min(float(aa.max()), float(bb.max()))
    if low > high + epsilon:
        return None
    point = sa[0] + direction * ((low + high) / 2 - float(sa[0] @ direction))
    bothStraddled = all(float(values.min()) < -epsilon and float(values.max()) > epsilon for values in (da, db))
    return {"kind": "nonCoplanarSegment" if high - low > epsilon else "pointContact",
            "point": point.tolist(), "segmentLength": max(0., high - low),
            "bothPlanesStraddled": bothStraddled}



def intersectionPairs(vertices, faces, *, epsilon=None):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if epsilon is None:
        epsilon = max(float(np.linalg.norm(np.ptp(vertices, axis=0))) * 1e-9, 1e-12)
    candidates, _ = broadPhase(vertices, faces, epsilon)
    triangles = vertices[faces]
    result = set()
    for first, second in candidates:
        hit = intersectTriangles(triangles[first], triangles[second], epsilon)
        if hit and hit['kind'] in ('nonCoplanarSegment', 'coplanarAreaOverlap'):
            result.add((int(first), int(second)))
    return result
