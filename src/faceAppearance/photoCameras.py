"""Camera-only conversion from saved UVFaceFusion/VGGT outputs to photo space."""

import numpy as np


def canonicalCamera(worldExtrinsics: np.ndarray, worldToCanonical: dict) -> dict:
    """C=s*(W@Q)+offset; return projection-equivalent proper R,t for C.

    World extrinsics use OpenCV column vectors Xcam=Rv*W+tv. Multiplying
    camera-space positions by positive s leaves pixel projections unchanged:
    Rc=Rv@Q and tc=s*tv-Rc@offset. No camera fitting takes place here.
    """
    extrinsics = np.asarray(worldExtrinsics, np.float64)
    rotation = np.asarray(worldToCanonical["R"], np.float64)
    offset = np.asarray(worldToCanonical["t"], np.float64)
    scale = float(worldToCanonical["s"])
    if extrinsics.shape != (3, 4) or rotation.shape != (3, 3) or offset.shape != (3,):
        raise ValueError("Invalid saved camera/canonicalization dimensions")
    if not all(np.isfinite(x).all() for x in (extrinsics, rotation, offset)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Camera transforms must be finite and similarity scale positive")
    for matrix in (extrinsics[:, :3], rotation):
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-4) or np.linalg.det(matrix) < 0.999:
            raise ValueError("Expected proper rotations, not an affine/reflection camera transform")
    canonicalRotation = extrinsics[:, :3] @ rotation
    return {"rotation": canonicalRotation,
            "translation": scale * extrinsics[:, 3] - canonicalRotation @ offset,
            "source": "uvFaceFusionVggtNative; exact canonical coordinate conversion only",
            "cameraDepthScale": scale}


def originalImageIntrinsics(workingIntrinsics: np.ndarray, inputToWorking: np.ndarray) -> np.ndarray:
    """Undo saved EXIF-oriented crop/resize pixel affine, including half-pixels."""
    intrinsics = np.asarray(workingIntrinsics, np.float64)
    affine = np.asarray(inputToWorking, np.float64)
    if intrinsics.shape != (3, 3) or affine.shape != (3, 3):
        raise ValueError("Expected 3x3 intrinsics and pixel affine")
    if not np.isfinite(intrinsics).all() or not np.isfinite(affine).all() or abs(np.linalg.det(affine)) < 1e-12:
        raise ValueError("Invalid pixel affine/intrinsics")
    return np.linalg.solve(affine, intrinsics)
