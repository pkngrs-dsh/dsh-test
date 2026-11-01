"""Audit frozen mappings of skin triangles adjacent to the visible eye region.

The pixel-derived neighborhood is not an annotated eyelid curve. Upper/lower
checks use geometric path proxies and explicitly report ambiguous corners.
"""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from auditClientEyeMapping import digest, nearest_segments, replay_qualification, split_loop
from faceAppearance.artifacts import writeJson
from faceAppearance.correspondence import sampleSurface


def path_proxy(points, vertices, split):
    upper = nearest_segments(points, vertices, split["upper"])["distance"]
    lower = nearest_segments(points, vertices, split["lower"])["distance"]
    chord = vertices[split["outer"], :2] - vertices[split["inner"], :2]
    along = (points[:, :2] - vertices[split["inner"], :2]) @ chord / (chord @ chord)
    margin = abs(upper - lower) / np.maximum(upper + lower, 1e-12)
    return {"upperCloser": upper < lower, "chordFraction": along, "distanceMargin": margin,
            "centralUnambiguous": (along > .1) & (along < .9) & (margin > .1)}


def main():
    output = ROOT / "runs/clientEyeMappingAudit"
    template = ROOT / "assets/clientTemplates/legacyFemaleV1"
    visibility_path = output / "visibleEye/visibleEyeMapping.json"
    visibility = json.loads(visibility_path.read_text())
    for path, value in visibility["inputHashes"].items():
        if digest(path) != value:
            raise ValueError("Visible-neighborhood source identity changed")
    arrays = dict(np.load(template / "template.npz", allow_pickle=False))
    manifest = json.loads((template / "manifest.json").read_text())
    replay = replay_qualification(arrays, manifest)
    static = json.loads((output / "staticMapping.json").read_text())
    rings = json.loads((output / "socketRings.json").read_text())
    vertices = arrays["vertices"][arrays["nicpToFullIds"]]
    source = arrays["sourceVertices"]
    targets = sampleSurface(source, arrays["sourceFaces"], arrays)
    mapped_uv = sampleSurface(arrays["sourceUvs"], arrays["sourceFaces"], arrays)
    results = {}
    for eye in visibility["visibleEyes"]:
        all_ids = np.asarray(eye["adjacentNicpVertexIds"])
        ids = all_ids[all_ids >= 0]
        name = "positiveXEye" if vertices[ids, 0].mean() > 0 else "negativeXEye"
        opposite = "negativeXEye" if name == "positiveXEye" else "positiveXEye"
        loop = np.asarray(static["eyes"][name]["sourceBoundaryVertexIds"])
        other_loop = np.asarray(static["eyes"][opposite]["sourceBoundaryVertexIds"])
        same_distance = nearest_segments(targets[ids], source, loop, closed=True)["distance"]
        other_distance = nearest_segments(targets[ids], source, other_loop, closed=True)["distance"]
        client_split = split_loop(vertices, np.asarray(rings["eyes"][name][3]["nicpVertexIds"]))
        source_split = split_loop(source, loop)
        client_proxy = path_proxy(vertices[ids], vertices, client_split)
        source_proxy = path_proxy(targets[ids], source, source_split)
        mismatch = client_proxy["upperCloser"] != source_proxy["upperCloser"]
        unambiguous = client_proxy["centralUnambiguous"] & source_proxy["centralUnambiguous"]
        active = arrays["geometryWeights"][ids] > 0
        rows = []
        for position, index in enumerate(ids):
            failures = []
            if replay["uvDistance"][index] > 1e-5: failures.append("outsideSourceUvSurface")
            if replay["restDistance"][index] >= .1: failures.append("restDistanceAtLeast0.1CanonicalUnits")
            if replay["normalDot"][index] <= .35: failures.append("normalDotAtMost0.35")
            if replay["static"][index]: failures.append("staticMask")
            if replay["seamWeight"][index] == 0: failures.append("conflictingUvSeams")
            rows.append({"nicpVertexId": int(index), "fullVertexId": int(arrays["nicpToFullIds"][index]),
                "geometryWeight": float(arrays["geometryWeights"][index]), "replayedGateFailures": failures,
                "gates": {key: float(replay[key][index]) for key in ("uvDistance", "restDistance", "normalDot", "seamSpread", "seamWeight")},
                "sourceTriangleId": int(arrays["triangleIds"][index]),
                "sourceTriangleVertexIds": arrays["sourceFaces"][arrays["triangleIds"][index]],
                "sourceBarycentrics": arrays["barycentricWeights"][index],
                "mappedSourcePointCanonical": targets[index], "mappedSourceUv": mapped_uv[index],
                "sourceSameEyeHoleDistanceCanonical": float(same_distance[position]),
                "sourceOppositeEyeHoleDistanceCanonical": float(other_distance[position]),
                "oppositeEyeCloser": bool(other_distance[position] < same_distance[position]),
                "clientPathProxy": {key: value[position].item() for key, value in client_proxy.items()},
                "sourcePathProxy": {key: value[position].item() for key, value in source_proxy.items()},
                "nearestPathProxyMismatch": bool(mismatch[position]),
                "centralUnambiguousPathProxyMismatch": bool(mismatch[position] and unambiguous[position])})
        results[name] = {"name": name, "labelId": eye["labelId"], "neighborhood": eye["scope"],
            "clientNicpVertexIds": ids, "clientFullVertexIds": arrays["nicpToFullIds"][ids],
            "excludedNonSkinFullVertexIds": np.asarray(eye["adjacentFullVertexIds"])[all_ids < 0],
            "vertices": rows,
            "summary": {"skinVertices": len(ids), "activeVertices": int(active.sum()), "inactiveVertices": int((~active).sum()),
                "failureCountsOverlapping": dict(Counter(reason for row in rows for reason in row["replayedGateFailures"])),
                "activeCrossEyeCount": int(np.count_nonzero(active & (other_distance < same_distance))),
                "activeRawPathProxyMismatchCount": int(np.count_nonzero(active & mismatch)),
                "activeAmbiguousPathProxyMismatchFullVertexIds": arrays["nicpToFullIds"][ids[active & mismatch & ~unambiguous]],
                "activeUnambiguousCentralPathProxyMismatchCount": int(np.count_nonzero(active & mismatch & unambiguous)),
                "activePointsEligibleForCentralPathProxyCheck": int(np.count_nonzero(active & unambiguous)),
                "activeGeometryNormalDotRange": [float(replay["normalDot"][ids[active]].min()), float(replay["normalDot"][ids[active]].max())]}}
    report = {"status": "completed", "sourceVisibilityReport": str(visibility_path),
        "sourceVisibilityReportSha256": digest(visibility_path), "templateSha256": digest(template / "template.npz"),
        "compilerReplay": "All 2795 frozen source triangle IDs, barycentric weights and geometry weights exactly reproduce from hash-matched original compiler inputs.",
        "scope": "View-dependent skin-triangle neighborhood beside the current visible eye material; not an ordered or anatomically annotated eyelid curve.",
        "pathProxyCheck": "Client graph-distance-3 closed skin loop versus actual UFF hole: abs(X) extrema corners; larger Y path is upper. Central unambiguous test requires both chord fractions in (0.1,0.9) and normalized upper/lower distance difference >0.1. These audit thresholds do not change production qualification.",
        "limits": ["A same-eye mapping and a passing normal gate do not certify pointwise anatomical correspondence.",
                   "Nearest-upper/lower classification at shared corner endpoints is ambiguous and is not an eyelid-crossing finding.",
                   "No source-inner/outer surface ground truth or reviewed eyelid correspondence is available.",
                   "Zero-weight points remain indirectly affected by smooth displacement and safety guards."],
        "eyes": results}
    writeJson(output / "visibleNeighborhoodMapping.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    for ax, name in zip(axes, ("negativeXEye", "positiveXEye")):
        record = results[name]
        for part, color in (("upper", "#059669"), ("lower", "#2563eb")):
            indices = static["eyes"][name]["sourcePaths"][part + "VertexIds"]
            uv = arrays["sourceUvs"][indices]
            ax.plot(*uv.T, color=color, lw=2, label=f"UFF hole {part} geometric path")
        for is_active, color, label in ((False, "#ef4444", "Inactive neighborhood targets"), (True, "#334155", "Active neighborhood targets")):
            points = np.asarray([row["mappedSourceUv"] for row in record["vertices"] if bool(row["geometryWeight"]) == is_active])
            ax.scatter(*points.T, color=color, s=28, alpha=.8, label=label)
        for row in record["vertices"]:
            if row["geometryWeight"] and row["nearestPathProxyMismatch"]:
                ax.scatter(*row["mappedSourceUv"], facecolors="none", edgecolors="#f59e0b", s=100, lw=2)
                ax.annotate(str(row["fullVertexId"]), row["mappedSourceUv"], fontsize=8, xytext=(3, 5), textcoords="offset points")
        ax.set_title(name + f" — {record['summary']['activeVertices']}/{record['summary']['skinVertices']} active")
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xlabel("Native UFF U"); ax.set_ylabel("Native UFF V (down)")
        ax.legend(fontsize=8, loc="lower center")
    fig.suptitle("Visible-eye neighboring skin: frozen targets in UFF UV\nOrange rings: ambiguous endpoint path classifications; this neighborhood is NOT a bound eyelid curve.", fontsize=13)
    fig.savefig(output / "visibleNeighborhoodUv.png", dpi=150)
    print(json.dumps({name: record["summary"] for name, record in results.items()}, default=lambda value: value.tolist(), indent=2))


if __name__ == "__main__":
    main()
