"""Audit saved camera/head equivalent poses without refitting or changing geometry."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faceAppearance.cameraDiagnostics import compareSavedCameras
from faceAppearance.photoCameras import canonicalCamera
from faceAppearance.artifacts import writeJson


def angle(rotation):
    # Nearest proper rotation removes saved float32 round-off before acos.
    u, _, vt = np.linalg.svd(rotation)
    corrected = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    return float(np.degrees(np.arccos(np.clip((np.trace(corrected) - 1) / 2, -1, 1))))


def relative(rotation, reference):
    return np.asarray(rotation) @ np.asarray(reference).T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    batch = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = {"status": "complete", "cases": [],
        "sourceManifestSha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "implementationSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conventions": {
            "worldToCanonicalRow": "C = s * (W @ Q) + o",
            "worldToCameraColumn": "x = Rv @ W + tv",
            "equivalentCanonicalCamera": "Rc = Rv @ Q; tc = s * tv - Rc @ o",
            "cameraPoseInHeadFrame": "cameraToHeadRotation = Rc.T; center = -Rc.T @ tc",
            "relativeRotation": "Rc_i @ Rc_0.T; maps reference camera-frame directions to view i for a fixed canonical head",
            "relativeTranslation": "tc_i - (Rc_i @ Rc_0.T) @ tc_0; uncalibrated canonical units",
            "physicalAmbiguity": "Effective head-to-camera transform E_i = inverse(CameraToWorld_i) @ HeadToWorld_i; face evidence alone does not uniquely separate physical camera motion from head motion",
        },
        "limits": ["Saved learned UV and depth predictions are not pose or geometry ground truth",
                   "Canonical conversion pixel error checks algebraic self-consistency only, not actual motion-estimation accuracy",
                   "Native-versus-PnP angular difference compares two estimates, not either estimate against ground truth",
                   "Reported angles are unsigned rotation magnitudes in [0,180] degrees, not signed yaw or pitch",
                   "VGGT world motion is model output, not an independently measured physical camera trajectory",
                   "A head turn can be expressed as an inverse virtual-camera motion only with the full rigid transform and the chosen pivot; Euler-angle negation alone is insufficient",
                   "Equivalent camera conversion cannot fix erroneous depth, expression changes, occlusion or changing illumination",
                   "Current UFF already canonicalizes each partial face separately; no second rotation correction was applied",
                   "Camera center units inherit uncalibrated canonical scale and are not metres"],
    }
    for case in batch["cases"]:
        if case["status"] != "succeeded":
            report["cases"].append({"caseId": case["caseId"], "status": "unavailable", "reason": case.get("error", case["status"])})
            continue
        directory = Path(case["outputDirectory"])
        with np.load(directory / "geometry.npz", allow_pickle=False) as saved:
            geometry = dict(saved)
        transforms = json.loads((directory / "viewTransforms.json").read_text())
        fitted = json.loads((directory / "cameras.json").read_text())
        native = [{**canonicalCamera(extrinsic, transform), "intrinsics": intrinsic}
                  for extrinsic, transform, intrinsic in zip(geometry["cameraExtrinsics"], transforms, geometry["cameraIntrinsics"])]
        comparisons = compareSavedCameras(geometry, native, fitted, seed=batch["config"]["seed"])
        views = []
        vertices = np.asarray(geometry["vertices"], np.float64)
        for i, (transform, camera, fitted_camera, comparison) in enumerate(zip(transforms, native, fitted, comparisons)):
            q, offset, scale = np.asarray(transform["R"], np.float64), np.asarray(transform["t"], np.float64), float(transform["s"])
            extrinsic = np.asarray(geometry["cameraExtrinsics"][i], np.float64)
            rc, tc = camera["rotation"], camera["translation"]
            # Use the actual inverse (not transpose) so the identity checks also
            # expose float32 orthogonality error in stored Q.
            world = ((vertices - offset) / scale) @ np.linalg.inv(q)
            through_world = scale * (world @ extrinsic[:, :3].T + extrinsic[:, 3])
            direct = vertices @ rc.T + tc
            intrinsics = np.asarray(camera["intrinsics"])
            p1, p2 = through_world @ intrinsics.T, direct @ intrinsics.T
            valid = (p1[:, 2] > 1e-8) & (p2[:, 2] > 1e-8)
            pixels1, pixels2 = p1[valid, :2] / p1[valid, 2:], p2[valid, :2] / p2[valid, 2:]
            view = {
                "viewIndex": i, "canonicalScale": scale,
                "canonicalizationSupportVertices": int(np.count_nonzero(transform["validVertexMask"])),
                "vggtWorldRelativeAngleDegrees": angle(relative(extrinsic[:, :3], geometry["cameraExtrinsics"][0, :, :3])),
                "canonicalizationRelativeAngleDegrees": angle(relative(q, transforms[0]["R"])),
                "nativeEffectiveRelativeAngleDegrees": angle(relative(rc, native[0]["rotation"])),
                "nativeEffectiveRotation": rc.tolist(), "nativeEffectiveTranslation": tc.tolist(),
                "nativeCameraToHeadRotation": rc.T.tolist(),
                "nativeCameraCenterCanonical": (-rc.T @ tc).tolist(),
                "nativeRelativeRotation": relative(rc, native[0]["rotation"]).tolist(),
                "nativeRelativeTranslation": (tc - relative(rc, native[0]["rotation"]) @ native[0]["translation"]).tolist(),
                "canonicalConversionMaxCoordinateError": float(np.max(np.abs(through_world - direct))),
                "canonicalConversionMaxPixelError": float(np.max(np.abs(pixels1 - pixels2))) if valid.any() else None,
                "canonicalRotationOrthogonalityError": float(np.max(np.abs(q.T @ q - np.eye(3)))),
                "fittedCameraValid": bool(fitted_camera.get("valid")),
                "uvConsistency": comparison,
            }
            if "rotation" in fitted_camera:
                rf, tf = np.asarray(fitted_camera["rotation"]), np.asarray(fitted_camera["translation"])
                view.update(fittedEffectiveRotation=rf.tolist(), fittedEffectiveTranslation=tf.tolist(),
                    fittedCameraToHeadRotation=rf.T.tolist(), fittedCameraCenterCanonical=(-rf.T @ tf).tolist(),
                    nativeVsFittedAngleDegrees=angle(relative(rc, rf)))
                if "rotation" in fitted[0]:
                    view.update(fittedEffectiveRelativeAngleDegrees=angle(relative(rf, fitted[0]["rotation"])),
                                fittedRelativeRotation=relative(rf, fitted[0]["rotation"]).tolist(),
                                fittedRelativeTranslation=(tf - relative(rf, fitted[0]["rotation"]) @ np.asarray(fitted[0]["translation"])).tolist())
            views.append(view)
        scales = [v["canonicalScale"] for v in views]
        item = {"caseId": case["caseId"], "label": case["label"], "status": "measured",
            "captureModeHypothesis": case.get("captureModeHypothesis"), "views": views,
            "scaleMaxOverMin": max(scales) / min(scales),
            "sourceHashes": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                             for name in ("geometry.npz", "cameras.json", "viewTransforms.json", "inputs.json")}}
        report["cases"].append(item)
        print(case["caseId"], "native/PnP angle deg:", [round(v.get("nativeVsFittedAngleDegrees", float("nan")), 2) for v in views], flush=True)
    writeJson(args.manifest.parent / "rotationDiagnostics.json", report)


if __name__ == "__main__":
    main()
