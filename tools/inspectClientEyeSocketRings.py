"""Inspect connected skin rings in front of the client's internal eye holes.

Graph distance defines geometric bands only, not anatomical eyelid landmarks.
No asset changes, model inference or rendering visibility claims are made.
"""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from faceAppearance.anatomicalBoundaries import extractMeshBoundaryLoops
from faceAppearance.artifacts import writeJson
from faceAppearance.correspondence import sampleSurface
from faceAppearance.texture import vertexNormals
from auditClientEyeMapping import digest, nearest_segments, replay_qualification


def main():
    path = ROOT / "assets/clientTemplates/legacyFemaleV1"
    output = ROOT / "runs/clientEyeMappingAudit"
    arrays = dict(np.load(path / "template.npz", allow_pickle=False))
    manifest = json.loads((path / "manifest.json").read_text())
    previous = json.loads((output / "staticMapping.json").read_text())
    replay = replay_qualification(arrays, manifest)
    vertices = arrays["vertices"][arrays["nicpToFullIds"]]
    faces = arrays["nicpFaces"]
    normals = vertexNormals(vertices, faces)
    targets = sampleSurface(arrays["sourceVertices"], arrays["sourceFaces"], arrays)
    edges = set(map(tuple, np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)))
    adjacency = [set() for _ in vertices]
    for first, second in edges:
        adjacency[first].add(second); adjacency[second].add(first)
    eyes = {}
    for loop in extractMeshBoundaryLoops(vertices, faces)["loops"][1:]:
        name = "positiveXEye" if vertices[loop, 0].mean() > 0 else "negativeXEye"
        ring, seen, records = set(loop), set(loop), []
        source_loop = np.asarray(previous["eyes"][name]["sourceBoundaryVertexIds"])
        for distance in range(7):
            local_adjacency = {index: adjacency[index] & ring for index in ring}
            if not all(len(value) == 2 for value in local_adjacency.values()):
                raise ValueError("Graph-distance band is not a simple mesh-edge loop")
            ids = [min(ring)]
            current = min(local_adjacency[ids[0]])
            while current != ids[0]:
                if current in ids:
                    raise ValueError("Unexpected graph ring cycle")
                ids.append(current)
                current = next(value for value in local_adjacency[current] if value != ids[-2])
            if len(ids) != len(ring):
                raise ValueError("Graph-distance band has multiple disconnected loops")
            ids = np.asarray(ids, np.int64)
            points = vertices[ids]
            source_distances = nearest_segments(targets[ids], arrays["sourceVertices"], source_loop, closed=True)["distance"]
            records.append({"graphDistanceFromInternalHole": distance, "nicpVertexIds": ids,
                "fullVertexIds": arrays["nicpToFullIds"][ids], "restPositionsCm": points,
                "restVertexNormals": normals[ids], "mappedSourcePointsCanonical": targets[ids],
                "geometryWeights": arrays["geometryWeights"][ids], "staticMask": arrays["staticMask"][ids],
                "sourceTriangleIds": arrays["triangleIds"][ids], "sourceBarycentrics": arrays["barycentricWeights"][ids],
                "qualification": {key: replay[key][ids] for key in ("uvDistance", "restDistance", "normalDot", "seamWeight")},
                "summary": {"vertexCount": len(ids), "activeGeometryConstraints": int(np.count_nonzero(arrays["geometryWeights"][ids])),
                    "closedMeshEdgeLoop": True, "restMeanZCm": float(points[:, 2].mean()),
                    "restZRangeCm": [float(points[:, 2].min()), float(points[:, 2].max())],
                    "restHeightCm": float(np.ptp(points[:, 1])), "restWidthCm": float(np.ptp(points[:, 0])),
                    "meanForwardNormalComponent": float(normals[ids, 2].mean()),
                    "mappedDistanceToUffHoleCanonicalMedianMax": np.percentile(source_distances, [50, 100])}})
            ring = set.union(*(adjacency[index] for index in ring)) - seen
            seen.update(ring)
        eyes[name] = records
    report = {"status": "completed", "templateHash": digest(path / "template.npz"), "eyes": eyes,
        "meaning": "Closed skin edge loops selected by graph distance from the internal topological eye hole; ring IDs are geometry diagnostics, not anatomical labels.",
        "finding": "The ten-vertex hole lies behind progressively forward-folded skin rings. Its larger opening and low geometry weights cannot be used as measurements of the visible eyelid.",
        "units": "Template centimeters, not calibrated patient measurements.", "visibilityAssessed": False,
        "legacyAnnotationCandidate": {"path": "/path/to/legacy-template/template_tri_lms_468_bicentroid.json",
             "status": "Unverified candidate: 468 face+barycentric entries consumed against registered NICP mesh in vendor/main_v1.ipynb cells 11/12 and vendor/v1_utils/custom_mesh/landmark_calculator.py. Topology applicability and eye landmark binding not audited here."}}
    writeJson(output / "socketRings.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    colors = ["#dc2626", "#f97316", "#eab308", "#16a34a", "#0891b2", "#2563eb", "#7c3aed"]
    for row, (name, records) in enumerate(eyes.items()):
        for record, color in zip(records, colors):
            points = np.asarray(record["restPositionsCm"])
            points = np.vstack([points, points[0]])
            count = record["summary"]["activeGeometryConstraints"]
            label = f"Ring {record['graphDistanceFromInternalHole']} ({count}/{len(points)-1} active)"
            axes[row, 0].plot(points[:, 0], points[:, 1], color=color, marker=".", label=label)
            axes[row, 1].plot(points[:, 2], points[:, 1], color=color, marker=".", label=label)
        axes[row, 0].set_title(name + " — frontal coordinates")
        axes[row, 1].set_title(name + " — depth coordinates (all X projected)")
        axes[row, 0].set_xlabel("Template X (cm)"); axes[row, 1].set_xlabel("Template Z (cm; forward →)")
        for ax in axes[row]:
            ax.set_ylabel("Template Y (cm)"); ax.set_aspect("equal"); ax.grid(alpha=.2)
        axes[row, 0].legend(fontsize=8)
    fig.suptitle("The 10-point internal hole is behind the folded skin\nGraph-distance rings are geometric diagnostics; no visible-eyelid annotation is assumed.", fontsize=14)
    fig.savefig(output / "socketRings.png", dpi=150)
    print(json.dumps({name: [record["summary"] for record in records] for name, records in eyes.items()}, default=lambda value: value.tolist()))


if __name__ == "__main__":
    main()
