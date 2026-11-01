"""Reproducible numeric and visual audit of a returned Wrap registration."""

import json
import sys
from pathlib import Path

import numpy as np

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot / "src"))
from prepareWrapBundle import readObj
from faceAppearance.artifacts import writeJson
from faceAppearance.correspondence import SurfaceIndex, qualifyMapping
from faceAppearance.texture import vertexNormals


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    output = projectRoot / "reports/registration"
    output.mkdir(parents=True, exist_ok=True)
    fixed = readObj(projectRoot / "assets/wrapPreparation/uvFaceFusionFixed.obj")
    registered = readObj(projectRoot / "assets/ictRegisteredToUvFaceFusion.obj")
    mappings, summary = {}, {}
    for name, target, source in (("ictToUff", registered, fixed), ("uffToIct", fixed, registered)):
        index = SurfaceIndex(source.vertices, source.faces)
        mapping = qualifyMapping(index.query(target.vertices), index,
                                 vertexNormals(target.vertices, target.faces), 0.01, 0.5)
        mappings[name] = mapping
        np.savez_compressed(output / f"{name}Vertices.npz", **mapping)
        summary[name] = {"queryVertexCount": len(target.vertices),
                         "validVertexCount": int(mapping["validMask"].sum()),
                         "distancePercentiles": dict(zip(["p50", "p90", "p95", "p99", "max"],
                             np.percentile(mapping["distances"], [50, 90, 95, 99, 100]).tolist())),
                         "invalidVertexIds": np.flatnonzero(~mapping["validMask"]).tolist(),
                         "distanceLimit": 0.01, "minNormalDot": 0.5}
    summary["registrationAccepted"] = False
    summary["note"] = "Numeric proximity is not anatomical correctness; source-template units, not millimeters."
    writeJson(output / "surfaceAudit.json", summary)
    figure, axes = plt.subplots(2, 3, figsize=(16, 12))
    views = [(0, 1, (-0.8, 0.8), (-1.06, 0.86), "Front: X/Y"),
             (2, 1, (-0.75, 0.4), (-1.06, 0.86), "Right side: Z/Y"),
             (0, 2, (-0.8, 0.8), (-0.75, 0.4), "Top: X/Z")]
    for column, (hAxis, vAxis, hBounds, vBounds, title) in enumerate(views):
        for mesh, color in ((fixed, "#159caa"), (registered, "#ba4c7c")):
            triangles = mesh.vertices[mesh.faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            lookingAxis = ({0, 1, 2} - {hAxis, vAxis}).pop()
            visible = normals[:, lookingAxis] > 0
            triangles = triangles[visible][..., [hAxis, vAxis]]
            segments = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
            axes[0, column].add_collection(LineCollection(segments, colors=color, linewidths=0.25, alpha=0.65))
        mapping = mappings["ictToUff"]
        colors = np.minimum(mapping["distances"], 0.02)
        scatter = axes[1, column].scatter(registered.vertices[:, hAxis], registered.vertices[:, vAxis],
                                          c=colors, s=4, cmap="viridis", vmin=0, vmax=0.02)
        invalid = ~mapping["validMask"]
        axes[1, column].scatter(registered.vertices[invalid, hAxis], registered.vertices[invalid, vAxis],
                                s=9, c="#dd3344", marker="x", linewidths=0.5)
        for row in range(2):
            axes[row, column].set_xlim(*hBounds)
            axes[row, column].set_ylim(*vBounds)
            axes[row, column].set_aspect("equal")
            axes[row, column].set_title(title if row == 0 else title + " / distance, red = excluded")
    figure.suptitle("Registration audit: cyan = UVFaceFusion, magenta = registered ICT\n"
                    "Distances in uncalibrated template units; no semantic approval implied")
    figure.colorbar(scatter, ax=axes[1, :].tolist(), shrink=0.7, label="ICT to fixed surface distance")
    figure.savefig(output / "registrationOverview.png", dpi=150)
    plt.close(figure)
    # Close-ups retain only the visible front-facing triangles.
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    regions = [((-0.5, 0.5), (0.12, 0.42), "Eyelids"),
               ((-0.22, 0.22), (-0.22, 0.18), "Nose"),
               ((-0.28, 0.28), (-0.45, -0.2), "Lips")]
    for ax, (xb, yb, label) in zip(axes, regions):
        for mesh, color in ((fixed, "#159caa"), (registered, "#ba4c7c")):
            xyz = mesh.vertices[mesh.faces]
            normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
            triangles = xyz[normals[:, 2] > 0][..., :2]
            segments = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
            ax.add_collection(LineCollection(segments, colors=color, linewidths=0.65, alpha=0.8))
        ax.set_xlim(*xb); ax.set_ylim(*yb); ax.set_aspect("equal"); ax.set_title(label)
    figure.suptitle("Front-facing mesh edges: UVFaceFusion cyan / registered ICT magenta")
    figure.savefig(output / "registrationCloseups.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({name: {"valid": s["validVertexCount"], "count": s["queryVertexCount"]}
                      for name, s in summary.items() if isinstance(s, dict)}))


if __name__ == "__main__":
    main()
