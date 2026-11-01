"""Audit frozen client eye-hole boundary correspondences without asset changes.

The upper/lower and corner split is a rest-template coordinate proxy: X
extrema define inner/outer corners and larger Y identifies the upper path.
Actual topology holes, source hashes, UV mapping and qualification are checked.
The client hole is an internal opening behind several forward-folded skin
rings; its boundary is not the visible eyelid contour. This audit must not be
interpreted as a visible-eyelid anatomical correspondence certificate.
No detector, geometry fit, model inference, or correspondence repair is run.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faceAppearance.anatomicalBoundaries import extractMeshBoundaryLoops, validateBoundaryAsset
from faceAppearance.artifacts import writeJson
from faceAppearance.clientTemplate import readLegacyObj
from faceAppearance.correspondence import SurfaceIndex, sampleSurface
from faceAppearance.texture import vertexNormals
from prepareClientTemplate import similarity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_loop(vertices, loop):
    """Return X-extrema corner proxies and connected paths in native Y-up."""
    loop = np.asarray(loop, np.int64)
    inner = int(np.argmin(abs(vertices[loop, 0])))
    outer = int(np.argmax(abs(vertices[loop, 0])))

    def walk(step):
        ids, index = [loop[inner]], inner
        while index != outer:
            index = (index + step) % len(loop)
            ids.append(loop[index])
        return np.asarray(ids, np.int64)

    first, second = walk(1), walk(-1)
    upper, lower = (first, second) if vertices[first[1:-1], 1].mean() > vertices[second[1:-1], 1].mean() else (second, first)
    return {"inner": int(loop[inner]), "outer": int(loop[outer]), "upper": upper, "lower": lower}


def nearest_segments(points, vertices, path, closed=False):
    path = np.r_[path, path[0]] if closed else np.asarray(path)
    first, second = vertices[path[:-1]], vertices[path[1:]]
    delta = second - first
    position = np.clip(np.sum((points[:, None] - first[None]) * delta[None], -1) /
                       np.maximum(np.sum(delta * delta, -1), 1e-30), 0, 1)
    closest = first[None] + position[..., None] * delta[None]
    distances = np.linalg.norm(points[:, None] - closest, axis=-1)
    index = distances.argmin(1)
    return {"distance": distances[np.arange(len(points)), index],
            "point": closest[np.arange(len(points)), index],
            "edge": np.stack([path[index], path[index + 1]], axis=1),
            "fraction": position[np.arange(len(points)), index]}


def replay_qualification(arrays, manifest):
    """Replay the original compiler gates from hash-matched legacy sources."""
    sources = manifest["sources"]
    for key in ("nicpMesh", "nicpUvMapping", "staticFaceMask", "sourceMesh", "sourceUvs"):
        if digest(sources[key]) != manifest["templateHashes"][key]:
            raise ValueError(f"Legacy input hash changed: {key}")
    nicp, mapped, source = [readLegacyObj(sources[key]) for key in ("nicpMesh", "nicpUvMapping", "sourceMesh")]
    source_uvs = np.asarray(np.load(sources["sourceUvs"], allow_pickle=False), np.float32)
    if (not np.array_equal(nicp["vertices"], arrays["vertices"][arrays["nicpToFullIds"]])
            or not np.array_equal(nicp["faces"], arrays["nicpFaces"])
            or not np.array_equal(mapped["faces"], nicp["faces"])
            or not np.array_equal(source["vertices"], arrays["sourceVertices"])
            or not np.array_equal(source_uvs, arrays["sourceUvs"])):
        raise ValueError("Legacy identity mismatch")
    planar = np.c_[source_uvs, np.zeros(len(source_uvs))]
    triangles = planar[arrays["sourceFaces"]]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    valid_ids = np.flatnonzero(area > 1e-14)
    query = SurfaceIndex(planar, arrays["sourceFaces"][valid_ids]).query(np.c_[mapped["uvs"], np.zeros(len(mapped["uvs"]))])
    query["triangleIds"] = valid_ids[query["triangleIds"]]
    uv_points = sampleSurface(arrays["sourceVertices"], arrays["sourceFaces"], query)
    candidates = [[] for _ in nicp["vertices"]]
    for vertex, uv in zip(mapped["faces"].ravel(), mapped["uvFaces"].ravel()):
        candidates[vertex].append(int(uv))
    selected = np.empty(len(candidates), np.int64)
    seam_spread = np.zeros(len(candidates))
    for vertex, values in enumerate(candidates):
        values = np.unique(values)
        best = np.lexsort((values, query["distances"][values]))[0]
        selected[vertex] = values[best]
        valid = values[query["distances"][values] <= 1e-5]
        if len(valid) > 1:
            seam_spread[vertex] = np.linalg.norm(uv_points[valid] - uv_points[values[best]], axis=1).max()
    uv_distance = query["distances"][selected]
    triangles = query["triangleIds"][selected]
    barycentrics = query["barycentricWeights"][selected]
    if not np.array_equal(triangles, arrays["triangleIds"]) or not np.array_equal(barycentrics, arrays["barycentricWeights"]):
        raise ValueError("Frozen source triangle/barycentric mapping cannot be reproduced exactly")
    _, rotation, scale, _, rest_distance = similarity(nicp["vertices"], uv_points[selected], uv_distance <= 1e-5)
    normals = vertexNormals(arrays["sourceVertices"], arrays["sourceFaces"])
    mapped_normals = (normals[arrays["sourceFaces"][triangles]] * barycentrics[..., None]).sum(1)
    mapped_normals /= np.maximum(np.linalg.norm(mapped_normals, axis=1, keepdims=True), 1e-12)
    normal_dot = np.sum(mapped_normals * (vertexNormals(nicp["vertices"], nicp["faces"]) @ rotation), axis=1)
    static = np.zeros(len(nicp["vertices"]), bool)
    static[np.unique(nicp["faces"][json.loads(Path(sources["staticFaceMask"]).read_text())])] = True
    seam_weight = np.clip((.05 - seam_spread) / .04, 0, 1)
    weight = ((uv_distance <= 1e-5) & (rest_distance < .1) & (normal_dot > .35) & ~static) * seam_weight
    if not np.array_equal(weight.astype(np.float32), arrays["geometryWeights"]) or not np.array_equal(static, arrays["staticMask"]):
        raise ValueError("Frozen geometry qualification cannot be reproduced exactly")
    return {"selectedLegacyUvs": mapped["uvs"][selected], "uvDistance": uv_distance,
            "restDistance": rest_distance, "normalDot": normal_dot,
            "seamSpread": seam_spread, "seamWeight": seam_weight, "static": static,
            "referenceScale": scale}


def plot_uv(arrays, eyes, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), layout="constrained")
    uv = arrays["sourceUvs"]
    edges = np.unique(np.sort(np.concatenate([arrays["sourceFaces"][:, [0, 1]], arrays["sourceFaces"][:, [1, 2]], arrays["sourceFaces"][:, [2, 0]]]), axis=1), axis=0)
    for ax, (name, record) in zip(axes, eyes.items()):
        loop = np.asarray(record["sourceBoundaryVertexIds"])
        mapped = np.asarray([row["mappedSourceUv"] for row in record["vertices"]])
        legacy = np.asarray([row["legacyQueryUv"] for row in record["vertices"]])
        low, high = np.minimum(uv[loop].min(0), legacy.min(0)) - .012, np.maximum(uv[loop].max(0), legacy.max(0)) + .012
        eligible = ((uv[edges].mean(1) >= low) & (uv[edges].mean(1) <= high)).all(1)
        for edge in edges[eligible]:
            ax.plot(*uv[edge].T, color="#d5dce4", lw=.5, zorder=0)
        for path_name, color in (("upper", "#059669"), ("lower", "#2563eb")):
            path = np.asarray(record["sourcePaths"][path_name + "VertexIds"])
            ax.plot(*uv[path].T, color=color, lw=3, label=f"UFF hole: {path_name} path proxy")
        ax.plot(*np.vstack([mapped, mapped[0]]).T, color="#f97316", lw=1.2, alpha=.8, label="Frozen target sequence")
        for row, start, end in zip(record["vertices"], legacy, mapped):
            ax.plot([start[0], end[0]], [start[1], end[1]], color="#475569", lw=1)
            color = "#dc2626" if row["geometryWeight"] else "#64748b"
            ax.scatter(*end, color=color, s=60 if row["geometryWeight"] else 28, zorder=5)
            ax.scatter(*start, color=color, marker="x", s=45, zorder=6)
            ax.annotate(str(row["fullVertexId"]), end, xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlim(low[0], high[0]); ax.set_ylim(high[1], low[1]); ax.set_aspect("equal")
        ax.set_title(name + " — 1/10 hole-boundary constraints active")
        ax.set_xlabel("Native UFF U"); ax.set_ylabel("Native UFF V (down)")
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("Client internal eye-hole boundary in the UFF UV chart (NOT the visible eyelid)\nLabels: full vertex IDs; dots: frozen targets; crosses: queries; red: active, gray: rejected", fontsize=13)
    fig.savefig(output / "eyeUvMapping.png", dpi=160)
    plt.close(fig)


def audit(template, source, client, boundaries, output):
    output.mkdir(parents=True, exist_ok=True)
    arrays = dict(np.load(template / "template.npz", allow_pickle=False))
    manifest = json.loads((template / "manifest.json").read_text())
    geometry = dict(np.load(source / "geometry.npz", allow_pickle=False))
    final = dict(np.load(client / "clientMesh.npz", allow_pickle=False))
    diagnostics = json.loads((client / "diagnostics.json").read_text())
    boundary_asset = json.loads(boundaries.read_text())
    source_boundaries = validateBoundaryAsset(arrays["sourceVertices"], arrays["sourceFaces"], boundary_asset)
    if not np.array_equal(geometry["faces"], arrays["sourceFaces"]) or not np.array_equal(final["faces"], arrays["faces"]):
        raise ValueError("Run topology does not match the audited correspondence assets")
    rest = arrays["vertices"][arrays["nicpToFullIds"]]
    extracted = extractMeshBoundaryLoops(rest, arrays["nicpFaces"])
    if extracted["diagnostics"]["loopVertexCounts"] != [40, 10, 10]:
        raise ValueError("Unexpected client topology; do not assume the last two holes are eyes")
    replay = replay_qualification(arrays, manifest)
    reference = sampleSurface(arrays["sourceVertices"], arrays["sourceFaces"], arrays)
    current = sampleSurface(geometry["vertices"], arrays["sourceFaces"], arrays)
    mapped_uv = sampleSurface(arrays["sourceUvs"], arrays["sourceFaces"], arrays)
    transform = diagnostics["templateToUff"]
    final_uff = final["vertices"] / manifest["outputUnitScale"] @ np.asarray(transform["rotation"]) * transform["scale"] + np.asarray(transform["translation"])
    final_skin = final_uff[arrays["nicpToFullIds"]]
    eyes = {}
    for loop_index, ids in enumerate(extracted["loops"]):
        if loop_index == 0:
            continue
        name = "positiveXEye" if rest[ids, 0].mean() > 0 else "negativeXEye"
        source_loop = source_boundaries[name]
        other_name = "negativeXEye" if name == "positiveXEye" else "positiveXEye"
        split, source_split = split_loop(rest, ids), split_loop(arrays["sourceVertices"], source_loop)
        closest = nearest_segments(reference[ids], arrays["sourceVertices"], source_loop, closed=True)
        opposite = nearest_segments(reference[ids], arrays["sourceVertices"], source_boundaries[other_name], closed=True)
        uv_closest = nearest_segments(mapped_uv[ids], arrays["sourceUvs"], source_loop, closed=True)
        upper = nearest_segments(reference[ids], arrays["sourceVertices"], source_split["upper"])
        lower = nearest_segments(reference[ids], arrays["sourceVertices"], source_split["lower"])
        current_distance = nearest_segments(current[ids], geometry["vertices"], source_loop, closed=True)
        final_distance = nearest_segments(final_skin[ids], geometry["vertices"], source_loop, closed=True)
        rows = []
        for order, vertex in enumerate(ids):
            part = ("innerCorner" if vertex == split["inner"] else "outerCorner" if vertex == split["outer"]
                    else "upper" if vertex in split["upper"] else "lower")
            nearest_part = "upper" if upper["distance"][order] < lower["distance"][order] else "lower"
            failures = []
            if replay["uvDistance"][vertex] > 1e-5: failures.append("outsideSourceUvSurface")
            if replay["restDistance"][vertex] >= .1: failures.append("restDistanceAtLeast0.1CanonicalUnits")
            if replay["normalDot"][vertex] <= .35: failures.append("normalDotAtMost0.35")
            if replay["static"][vertex]: failures.append("staticMask")
            if replay["seamWeight"][vertex] == 0: failures.append("conflictingUvSeams")
            rows.append({"loopPosition": order, "nicpVertexId": int(vertex), "fullVertexId": int(arrays["nicpToFullIds"][vertex]),
                "partProxy": part, "sourceTriangleId": int(arrays["triangleIds"][vertex]),
                "sourceTriangleVertexIds": arrays["sourceFaces"][arrays["triangleIds"][vertex]],
                "barycentricWeights": arrays["barycentricWeights"][vertex],
                "geometryWeight": float(arrays["geometryWeights"][vertex]), "static": bool(arrays["staticMask"][vertex]),
                "replayedGateFailures": failures,
                "gates": {key: float(replay[key][vertex]) for key in ("uvDistance", "restDistance", "normalDot", "seamSpread", "seamWeight")},
                "legacyQueryUv": replay["selectedLegacyUvs"][vertex], "mappedSourceUv": mapped_uv[vertex],
                "mappedRestPointUff": reference[vertex], "mappedCurrentPointUff": current[vertex], "finalCurrentPointUff": final_skin[vertex],
                "nearestSameEyeRimEdge": closest["edge"][order], "nearestSameEyeRimEdgeFraction": float(closest["fraction"][order]),
                "restRimDistanceCanonical": float(closest["distance"][order]),
                "restRimDistanceDividedByReferenceScaleCm": float(closest["distance"][order] / replay["referenceScale"]),
                "sourceUvRimDistanceAt1024": float(uv_closest["distance"][order] * 1023),
                "oppositeEyeCloser": bool(opposite["distance"][order] < closest["distance"][order]),
                "nearestPathProxy": nearest_part,
                "upperDistanceCanonical": float(upper["distance"][order]), "lowerDistanceCanonical": float(lower["distance"][order]),
                "crossPathProxy": bool(part in ("upper", "lower") and nearest_part != part),
                "currentMappedRimDistanceCanonical": float(current_distance["distance"][order]),
                "currentFinalRimDistanceCanonical": float(final_distance["distance"][order]),
                "finalToMappedTargetResidualCm": float(np.linalg.norm(final_skin[vertex] - current[vertex]) / transform["scale"]),
                "deformationRetention": float(diagnostics["deformationRetention"][vertex])})
        duplicates = []
        for first in range(len(ids)):
            for second in range(first + 1, len(ids)):
                if np.array_equal(reference[ids[first]], reference[ids[second]]):
                    duplicates.append({"nicpVertexIds": ids[[first, second]], "fullVertexIds": arrays["nicpToFullIds"][ids[[first, second]]],
                                       "restClientDistanceCm": float(np.linalg.norm(rest[ids[first]] - rest[ids[second]])),
                                       "activeWeights": arrays["geometryWeights"][ids[[first, second]]]})
        eyes[name] = {"name": name, "clientLoopIndex": loop_index,
            "clientNicpVertexIds": ids, "clientFullVertexIds": arrays["nicpToFullIds"][ids],
            "sourceBoundaryVertexIds": source_loop,
            "clientCorners": {"innerNicpVertex": split["inner"], "outerNicpVertex": split["outer"],
                              "innerFullVertex": int(arrays["nicpToFullIds"][split["inner"]]), "outerFullVertex": int(arrays["nicpToFullIds"][split["outer"]])},
            "clientPaths": {**{part + "NicpVertexIds": split[part] for part in ("upper", "lower")},
                            **{part + "FullVertexIds": arrays["nicpToFullIds"][split[part]] for part in ("upper", "lower")}},
            "sourceCorners": {"innerVertex": source_split["inner"], "outerVertex": source_split["outer"]},
            "sourcePaths": {part + "VertexIds": source_split[part] for part in ("upper", "lower")},
            "vertices": rows, "duplicateTargets": duplicates,
            "summary": {"vertexCount": len(ids), "activeGeometryConstraints": int(np.count_nonzero(arrays["geometryWeights"][ids])),
                        "wrongEyeCount": sum(row["oppositeEyeCloser"] for row in rows),
                        "upperLowerProxyMismatchCount": sum(row["crossPathProxy"] for row in rows),
                        "sourceUvRimDistanceAt1024MedianMax": np.percentile(uv_closest["distance"] * 1023, [50, 100]),
                        "exactOnSourceRimCountTolerance1eMinus7": int(np.count_nonzero(closest["distance"] <= 1e-7)),
                        "gateFailureCounts": dict(Counter(reason for row in rows for reason in row["replayedGateFailures"])),
                        "allRetentionOne": bool(np.all(np.asarray(diagnostics["deformationRetention"])[ids] == 1))}}
    report = {"schemaVersion": 1, "status": "completed", "scope": "Frozen topological eye-hole boundary correspondence audit; no assets modified",
              "indexConvention": "All zero based. Client NICP indices address 2795 vertices; full indices address 3544 vertices; source indices address 6986 UFF vertices.",
              "naming": "Eye names are taken from reviewed UFF boundary annotations and matched to client rest X sign; anatomical left/right not assumed.",
              "partBinding": "Fixed rest-template abs(X) extrema are corner proxies; connected path with larger mean Y is upper. These labels are geometric proxies, not manually certified landmarks.",
              "clientBoundaryAnatomyVerified": False,
              "clientBoundaryMeaning": "The 10-vertex client topological hole is an internal opening behind forward-folded skin rings, not the visible eyelid silhouette. Its geometry weights and path mismatches describe that internal boundary only; do not extend them to all visible eyelid vertices.",
              "metricUnits": "UFF canonical units are uncalibrated. Distances divided by similarity scale express template centimeters, not measured patient millimeters.",
              "template": str(template.resolve()), "source": str(source.resolve()), "client": str(client.resolve()),
              "hashes": {str(path.resolve()): digest(path) for path in [template / "template.npz", template / "manifest.json", source / "geometry.npz", client / "clientMesh.npz", client / "diagnostics.json", boundaries]},
              "compilerReplay": {"sourceHashesMatch": True, "triangleIdsExactlyMatch": True, "barycentricWeightsExactlyMatch": True,
                                 "geometryWeightsExactlyMatch": True, "referenceScale": replay["referenceScale"]},
              "clientBoundaryDiagnostics": extracted["diagnostics"], "eyes": eyes}
    writeJson(output / "staticMapping.json", report)
    np.savez_compressed(output / "staticMapping.npz", **{f"{name}_{key}": value for name, record in eyes.items()
                       for key, value in {"clientNicpVertexIds": record["clientNicpVertexIds"], "clientFullVertexIds": record["clientFullVertexIds"],
                                          "sourceBoundaryVertexIds": record["sourceBoundaryVertexIds"],
                                          "mappedRestPointUff": reference[record["clientNicpVertexIds"]],
                                          "mappedCurrentPointUff": current[record["clientNicpVertexIds"]],
                                          "finalCurrentPointUff": final_skin[record["clientNicpVertexIds"]]}.items()})
    plot_uv(arrays, eyes, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=ROOT / "assets/clientTemplates/legacyFemaleV1")
    parser.add_argument("--source", type=Path, default=ROOT / "runs/service/b775794b61894a8b804fddb0620db45d/output")
    parser.add_argument("--client", type=Path, default=ROOT / "runs/sdfSkinTemplateBlendFinal/clientAssets/sdf")
    parser.add_argument("--boundaries", type=Path, default=ROOT / "assets/anatomical/uff6986/boundaries.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/clientEyeMappingAudit")
    args = parser.parse_args()
    result = audit(args.template, args.source, args.client, args.boundaries, args.output)
    print(json.dumps({name: record["summary"] for name, record in result["eyes"].items()}, default=lambda value: value.tolist(), indent=2))


if __name__ == "__main__":
    main()
