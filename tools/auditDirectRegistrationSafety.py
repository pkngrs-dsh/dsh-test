"""Read-only nonadjacent skin-triangle intersection audit for direct experiments.

Sweep-and-prune AABBs generate candidates. Plane clipping and interval overlap
find noncoplanar intersections; convex 2-D clipping handles coplanar overlap.
Triangles sharing any vertex index are deliberately excluded. This is a finite
precision geometric check, not an anatomical or collision-free certification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ("rest", "baseline", "legacyPrior", "weakPrior", "noPrior")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
            "point": point.tolist(), "overlapAreaCm2": float(area), "bothPlanesStraddled": False}


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
            "point": point.tolist(), "segmentLengthCm": max(0., high - low),
            "bothPlanesStraddled": bothStraddled}


def satIntersects(a, b, epsilon):
    """Independent separating-axis predicate, including coplanar edge axes."""
    ea, eb = np.roll(a, -1, axis=0) - a, np.roll(b, -1, axis=0) - b
    na, nb = np.cross(ea[0], ea[1]), np.cross(eb[0], eb[1])
    axes = np.concatenate([na[None], nb[None], np.cross(ea[:, None, :], eb[None, :, :]).reshape(-1, 3),
                           np.cross(na, ea), np.cross(nb, eb)])
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > epsilon * epsilon] / lengths[lengths > epsilon * epsilon, None]
    pa, pb = a @ axes.T, b @ axes.T
    return not ((pa.max(axis=0) < pb.min(axis=0) - epsilon)
                | (pb.max(axis=0) < pa.min(axis=0) - epsilon)).any()


def selfTest():
    first = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    crossing = np.array([[.5, .5, -1.], [.5, .5, 1.], [1.5, .5, 0.]])
    result = intersectTriangles(first, crossing, 1e-9)
    assert result and result["kind"] == "nonCoplanarSegment" and result["segmentLengthCm"] > .9
    assert intersectTriangles(first, crossing + [3, 0, 0], 1e-9) is None
    assert intersectTriangles(first, first + [0, 0, .01], 1e-9) is None
    inside = np.array([[.2, .2, 0.], [.8, .2, 0.], [.2, .8, 0.]])
    result = intersectTriangles(first, inside, 1e-9)
    assert result["kind"] == "coplanarAreaOverlap" and abs(result["overlapAreaCm2"] - .18) < 1e-9
    touching = np.array([[2., 0., 0.], [3., 0., 1.], [3., 1., 1.]])
    assert intersectTriangles(first, touching, 1e-9)["kind"] == "pointContact"
    assert intersectTriangles(first, first + [3, 0, 0], 1e-9) is None
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    pairs, excluded = broadPhase(vertices, np.array([[0, 1, 2], [0, 1, 3]]), 1e-9)
    assert len(pairs) == 0 and excluded == 1
    return {"status": "passed", "assertions": 7,
            "scope": "Synthetic crossing, separated, coplanar area, coplanar separation, point contact and shared-index exclusion"}


def regionMasks(rest, faces):
    centers = rest[faces].mean(axis=1)
    mouth = rest[[168, 1143]].mean(axis=0)
    oral = ((np.abs(centers[:, 0] - mouth[0]) < 3.5)
            & (np.abs(centers[:, 1] - mouth[1]) < 2.) & (centers[:, 2] > mouth[2] - 2.))
    deepOral = ((np.abs(centers[:, 0] - mouth[0]) < 3.5)
                & (np.abs(centers[:, 1] - mouth[1]) < 3.)
                & (centers[:, 2] <= mouth[2] - 2.) & (centers[:, 2] > mouth[2] - 8.))
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundaryVertices = np.unique(unique[counts == 1])
    # On this fixed NICP topology the ten-vertex loops above the mouth belong
    # to the eye openings. The large lower/back loop is the neck opening.
    eyeVertices = boundaryVertices[rest[boundaryVertices, 1] > mouth[1] + 2.]
    eye = np.zeros(len(faces), bool)
    eyeCenters = []
    for sign in (-1, 1):
        ids = eyeVertices[sign * (rest[eyeVertices, 0] - mouth[0]) > 0]
        if len(ids):
            center = rest[ids].mean(axis=0)
            eyeCenters.append(center.tolist())
            eye |= ((np.abs(centers[:, 0] - center[0]) < 2.5)
                    & (np.abs(centers[:, 1] - center[1]) < 2.) & (centers[:, 2] > center[2] - 1.5))
    return {"mouthRegion": oral, "deepMouthRegion": deepOral, "eyeOpeningRegion": eye}, {
        "mouthCenterRestCm": mouth.tolist(), "mouthFaceCount": int(oral.sum()),
        "deepMouthFaceCount": int(deepOral.sum()), "eyeFaceCount": int(eye.sum()),
        "eyeOpeningBoundaryVertexCount": len(eyeVertices),
        "eyeOpeningCentersRestCm": eyeCenters,
        "scope": "Rest-space broad spatial neighborhoods; not reviewed anatomical lip/eyelid labels"}


def frontVisibility(point, vertices, faces, tolerance=0.01):
    triangles = vertices[faces]
    xy, point = triangles[:, :, :2], np.asarray(point)
    candidate = ((xy.min(axis=1) <= point[:2] + 1e-9) & (xy.max(axis=1) >= point[:2] - 1e-9)).all(axis=1)
    tri = triangles[candidate]
    if not len(tri):
        return {"frontExposed": None, "occlusionDepthCm": None}
    e1, e2, delta = tri[:, 1, :2] - tri[:, 0, :2], tri[:, 2, :2] - tri[:, 0, :2], point[:2] - tri[:, 0, :2]
    cross = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    denominator = cross(e1, e2)
    usable = np.abs(denominator) > 1e-12
    denominator = np.where(usable, denominator, 1.)
    u, v = cross(delta, e2) / denominator, cross(e1, delta) / denominator
    inside = usable & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
    if not inside.any():
        return {"frontExposed": None, "occlusionDepthCm": None}
    depths = tri[:, 0, 2] + u * (tri[:, 1, 2] - tri[:, 0, 2]) + v * (tri[:, 2, 2] - tri[:, 0, 2])
    depth = max(0., float(depths[inside].max() - point[2]))
    return {"frontExposed": depth <= tolerance, "occlusionDepthCm": depth}


def auditMesh(vertices, faces, rest, masks, epsilon):
    started = perf_counter()
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    twiceArea = np.linalg.norm(cross, axis=1)
    edges2 = sum(np.sum((tri[:, i] - tri[:, j]) ** 2, axis=1) for i, j in ((0, 1), (1, 2), (2, 0)))
    quality = 2 * np.sqrt(3) * twiceArea / np.maximum(edges2, 1e-30)
    restTri = rest[faces]
    restTwiceArea = np.linalg.norm(np.cross(restTri[:, 1] - restTri[:, 0], restTri[:, 2] - restTri[:, 0]), axis=1)
    areaRatio = twiceArea / np.maximum(restTwiceArea, 1e-30)
    degenerate = twiceArea <= 2 * epsilon * epsilon
    skinny = quality < 1e-4
    reduced = areaRatio < .02
    pairs, excludedNeighbors = broadPhase(vertices, faces, epsilon)
    hits, predicateDisagreements = [], []
    for first, second in pairs:
        result = intersectTriangles(tri[first], tri[second], epsilon)
        sat = satIntersects(tri[first], tri[second], epsilon)
        if bool(result is not None) != bool(sat):
            predicateDisagreements.append({"faceIds": [int(first), int(second)],
                                           "planeClip": result is not None, "separatingAxes": bool(sat)})
        if result is None:
            continue
        result["faceIds"] = [int(first), int(second)]
        result["regions"] = [name for name, mask in masks.items() if mask[first] or mask[second]] or ["otherSkin"]
        result.update(frontVisibility(result["point"], vertices, faces))
        hits.append(result)
    summary = dict(Counter(hit["kind"] for hit in hits))
    summary["allPairs"] = len(hits)
    summary["positiveLengthOrAreaPairs"] = sum(hit["kind"] in ("nonCoplanarSegment", "coplanarAreaOverlap") for hit in hits)
    summary["bothPlanesStraddledPairs"] = sum(hit["bothPlanesStraddled"] for hit in hits)
    summary["regions"] = {region: {
        "allPairs": sum(region in hit["regions"] for hit in hits),
        "positiveLengthOrAreaPairs": sum(region in hit["regions"] and hit["kind"] in ("nonCoplanarSegment", "coplanarAreaOverlap") for hit in hits),
        "frontExposedPairs": sum(region in hit["regions"] and hit["frontExposed"] is True for hit in hits),
        "frontOccludedPairs": sum(region in hit["regions"] and hit["frontExposed"] is False for hit in hits),
    } for region in (*masks, "otherSkin")}
    return {"summary": summary, "intersections": hits, "broadPhasePairs": len(pairs),
            "independentSeparatingAxisCheck": {"checkedPairs": len(pairs), "disagreements": predicateDisagreements},
            "sharedVertexAabbPairsExcluded": excludedNeighbors,
            "degeneracy": {"zeroAreaFaceIds": np.flatnonzero(degenerate).tolist(),
                           "nearDegenerateShapeFaceIds": np.flatnonzero(skinny).tolist(),
                           "areaUnder2PercentRestFaceIds": np.flatnonzero(reduced).tolist(),
                           "minimumDimensionlessTriangleQuality": float(quality.min()),
                           "minimumRestAreaRatio": float(areaRatio.min())},
            "elapsedSeconds": perf_counter() - started}


def pairSet(branch):
    return {tuple(hit["faceIds"]) for hit in branch["intersections"]
            if hit["kind"] in ("nonCoplanarSegment", "coplanarAreaOverlap")}


def renderMouthLocations(directory, arrays, report, destination):
    from PIL import Image, ImageDraw, ImageFont
    from renderLegacyMouthAudit import render
    center = np.asarray(report["regionDefinition"]["mouthCenterRestCm"])
    bounds = (center[0] - 4, center[0] + 4, center[1] - 3, center[1] + 3)
    width, height = 420, 315
    names = list(report["branches"])
    result = Image.new("RGB", (width * len(names), height + 94), (28, 35, 39))
    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    draw.text((12, 5), directory.name + " | red: front exposed; blue: front occluded; white ring: new face pair vs baseline", font=font, fill="white")
    draw.text((12, 28), "Skin-only +Z view. Dots are intersection midpoints, not validated lip landmarks. Each pair can belong to one collision cluster.", font=font, fill="white")
    for column, name in enumerate(names):
        branch = report["branches"][name]
        panel = render(arrays[name], arrays["faces"], bounds, width)
        result.paste(panel, (column * width, 94))
        draw.text((column * width + 10, 58), f"{name}: {branch['summary']['positiveLengthOrAreaPairs']} pairs", font=font, fill="white")
        new = {tuple(pair) for pair in branch["comparison"]["newPositivePairsVsBaseline"]}
        for hit in branch["intersections"]:
            if not any(region in hit["regions"] for region in ("mouthRegion", "deepMouthRegion")):
                continue
            point = hit["point"]
            x = (point[0] - bounds[0]) / (bounds[1] - bounds[0]) * (width - 1)
            y = (bounds[3] - point[1]) / (bounds[3] - bounds[2]) * (height - 1)
            if not (0 <= x < width and 0 <= y < height):
                continue
            x += column * width
            y += 94
            fill = (255, 75, 60) if hit["frontExposed"] else (50, 160, 255)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=fill)
            if name != "rest" and tuple(hit["faceIds"]) in new:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline="white")
    result.save(destination)


def runCase(directory):
    meshPath = directory / "meshes.npz"
    with np.load(meshPath, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    rest, faces = arrays["rest"], arrays["faces"]
    if faces.shape != (5532, 3) or rest.shape != (2795, 3):
        raise ValueError("Expected the original 2795-vertex NICP skin only")
    if not np.isfinite(rest).all() or np.any(faces < 0) or np.any(faces >= len(rest)):
        raise ValueError("Invalid NICP geometry")
    masks, regionDefinition = regionMasks(rest, faces)
    epsilon = float(np.linalg.norm(np.ptp(rest, axis=0)) * 1e-9)
    branches = {}
    names = [*BRANCHES, *(name for name in arrays if name not in (*BRANCHES, "source", "sourceFaces", "faces"))]
    for name in names:
        if arrays[name].shape != rest.shape or not np.isfinite(arrays[name]).all():
            raise ValueError("Candidate vertex count or finite values differ")
        branches[name] = auditMesh(arrays[name], faces, rest, masks, epsilon)
        print(json.dumps({"case": directory.name, "branch": name, **branches[name]["summary"]}), flush=True)
    restPairs, baselinePairs = pairSet(branches["rest"]), pairSet(branches["baseline"])
    for name, branch in branches.items():
        actual = pairSet(branch)
        new = actual - baselinePairs
        branch["comparison"] = {
            "newPositivePairsVsRest": [list(pair) for pair in sorted(actual - restPairs)],
            "newPositivePairsVsBaseline": [list(pair) for pair in sorted(new)],
            "removedPositivePairsVsBaseline": [list(pair) for pair in sorted(baselinePairs - actual)],
            "newVsBaselineRegionCounts": dict(Counter(region for hit in branch["intersections"]
                if tuple(hit["faceIds"]) in new for region in hit["regions"])),
            "newVsBaselineMouthFrontExposed": sum(tuple(hit["faceIds"]) in new and "mouthRegion" in hit["regions"]
                                                   and hit["frontExposed"] is True for hit in branch["intersections"]),
            "newNearDegenerateShapeFacesVsRest": sorted(set(branch["degeneracy"]["nearDegenerateShapeFaceIds"])
                                                        - set(branches["rest"]["degeneracy"]["nearDegenerateShapeFaceIds"])),
        }
    source, sourceFaces = arrays["source"], arrays["sourceFaces"]
    centers = source[sourceFaces].mean(axis=1)
    mouth = np.asarray(regionDefinition["mouthCenterRestCm"])
    sourceMasks = {
        "mouthRegion": ((np.abs(centers[:, 0] - mouth[0]) < 3.5) & (np.abs(centers[:, 1] - mouth[1]) < 2.)
                        & (centers[:, 2] > mouth[2] - 2.)),
        "deepMouthRegion": ((np.abs(centers[:, 0] - mouth[0]) < 3.5) & (np.abs(centers[:, 1] - mouth[1]) < 3.)
                            & (centers[:, 2] <= mouth[2] - 2.) & (centers[:, 2] > mouth[2] - 8.)),
        "eyeOpeningRegion": np.zeros(len(sourceFaces), bool),
    }
    for center in np.asarray(regionDefinition["eyeOpeningCentersRestCm"]):
        sourceMasks["eyeOpeningRegion"] |= ((np.abs(centers[:, 0] - center[0]) < 2.5)
                                           & (np.abs(centers[:, 1] - center[1]) < 2.)
                                           & (centers[:, 2] > center[2] - 1.5))
    sourceAudit = auditMesh(source, sourceFaces, source, sourceMasks, epsilon)
    sourceAudit["comparisonScope"] = "Independent source topology; no face-ID set subtraction against NICP. Area ratios use source itself, so only absolute degeneracy and triangle quality are informative."
    sourceAudit["vertexCount"], sourceAudit["triangleCount"] = len(source), len(sourceFaces)
    print(json.dumps({"case": directory.name, "branch": "UFF source", **sourceAudit["summary"]}), flush=True)
    return {"caseId": directory.name, "meshSha256": _hash(meshPath), "skinVertexCount": len(rest),
            "skinTriangleCount": len(faces), "lengthEpsilonCm": epsilon,
            "regionDefinition": regionDefinition, "branches": branches, "sourceReference": sourceAudit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "runs/directRegistration20261012")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/directRegistration20261012/safety.json")
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    tests = selfTest()
    order = ("maleExactRegression", "orbitControl4", "highPitch4", "cyanWall4", "outdoorSideLight4", "backlitCap4")
    report = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(),
              "status": "running", "method": "AABB sweep; triangle/plane clipping; intersection-line interval overlap; coplanar convex polygon clipping",
              "selfTest": tests, "compilerSha256": _hash(__file__),
              "independentCheck": "Separating-axis test including plane normals, edge cross products and coplanar edge normals for every AABB candidate pair",
              "units": "experiment display template centimeters; physical scale is not calibrated",
              "thresholds": {"lengthEpsilon": "rest bounding-box diagonal * 1e-9", "parallelNormalsCrossNorm": 1e-10,
                             "zeroAreaCm2": "epsilon^2", "nearDegenerateDimensionlessQuality": 1e-4,
                             "areaFractionOfRest": .02, "frontDepthToleranceCm": .01},
              "limitations": [
                  "Pairs sharing any vertex index are excluded; adjacent-face foldovers are outside this particular self-intersection count.",
                  "Coplanar positive polygon area is overlap; zero-area contact is reported separately. Degenerate faces are reported independently.",
                  "Positive-length pairs and straddling flags describe geometry; rest-normal reversal is not used as an intersection predicate.",
                  "Region labels are rest-space neighborhoods, not reviewed visible lip-line or inner-mouth anatomy.",
                  "Front visibility is an orthographic +Z skin-only depth test at each intersection midpoint with 0.1 mm display-unit tolerance; hidden points suggest interior placement but are not anatomical certification.",
                  "No continuous motion, accessories, thickness, rigging or finite-clearance audit is performed.",
              ], "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in order:
        if args.case and case not in args.case:
            continue
        caseReport = runCase(args.input / case)
        with np.load(args.input / case / "meshes.npz", allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        plot = args.output.parent / f"{case}MouthIntersections.png"
        renderMouthLocations(args.input / case, arrays, caseReport, plot)
        caseReport["mouthIntersectionPlot"] = str(plot.resolve())
        report["cases"].append(caseReport)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    report["status"] = "completed_geometric_audit_not_anatomical_certification"
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
