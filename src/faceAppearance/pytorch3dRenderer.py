"""PyTorch3D rasterization with explicit tangent normals and a GGX preview shader.

PyTorch3D defines camera/view conventions, but not a normal-map baking standard.
The caller supplies OBJ-V-up TBN frames and signed positive-V tangent normals.
All lighting is linear; display output is sRGB. This diagnostic has no shadow
maps, SSS, environment transport or fitted roughness.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as functional
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (FoVOrthographicCameras, MeshRasterizer,
                               RasterizationSettings, TexturesUV, look_at_view_transform)
from pytorch3d.structures import Meshes


def linearToSrgb(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp_min(0)
    return torch.where(values <= 0.0031308, 12.92 * values,
                       1.055 * values.pow(1 / 2.4) - 0.055)


def applyTangentNormals(frames: torch.Tensor, tangentNormals: torch.Tensor) -> torch.Tensor:
    """Re-orthogonalize interpolated frames, preserving mirrored-UV handedness."""
    normals = functional.normalize(frames[..., 2], dim=-1)
    tangents = frames[..., 0]
    tangents = functional.normalize(tangents - (tangents * normals).sum(-1, keepdim=True) * normals, dim=-1)
    crossVectors = torch.cross(normals, tangents, dim=-1)
    handedness = torch.where((crossVectors * frames[..., 1]).sum(-1, keepdim=True) < 0, -1., 1.)
    bitangents = crossVectors * handedness
    return functional.normalize(tangents * tangentNormals[..., :1]
                                + bitangents * tangentNormals[..., 1:2]
                                + normals * tangentNormals[..., 2:3], dim=-1)


def shadeGgx(albedo: torch.Tensor, normals: torch.Tensor, viewDirections: torch.Tensor,
             specularStrength: torch.Tensor, lightDirection, *, roughness: float = 0.45,
             keyIntensity: float = 2.5, ambient: float = 0.18) -> torch.Tensor:
    """Linear Lambert + isotropic GGX/Smith/Schlick, with a constant ambient term.

    F0=0.08 and a separate specular multiplier follow the local WildCap
    shade_hyx preview's parameterization. Roughness is a chosen preview value.
    lightDirection points from the surface toward a directional light.
    """
    if not 0.05 <= roughness <= 1 or keyIntensity < 0 or ambient < 0:
        raise ValueError("Invalid preview lighting/roughness")
    light = functional.normalize(torch.as_tensor(lightDirection, device=normals.device,
                                                  dtype=normals.dtype), dim=-1)
    viewDirections = functional.normalize(viewDirections, dim=-1)
    halfVectors = functional.normalize(light + viewDirections, dim=-1)
    normalLight = (normals * light).sum(-1, keepdim=True).clamp(0, 1)
    normalView = (normals * viewDirections).sum(-1, keepdim=True).clamp(0, 1)
    normalHalf = (normals * halfVectors).sum(-1, keepdim=True).clamp(0, 1)
    viewHalf = (viewDirections * halfVectors).sum(-1, keepdim=True).clamp(0, 1)
    alphaSquared = roughness ** 4
    distribution = alphaSquared / (math.pi * (normalHalf.square() * (alphaSquared - 1) + 1).square())

    def smithTerm(cosine):
        return 2 * cosine / (cosine + torch.sqrt(alphaSquared + (1 - alphaSquared) * cosine.square())).clamp_min(1e-7)

    fresnel = 0.08 + 0.92 * (1 - viewHalf).pow(5)
    specular = distribution * smithTerm(normalLight) * smithTerm(normalView) * fresnel
    specular = specular / (4 * normalLight * normalView).clamp_min(1e-7)
    direct = ((1 - fresnel * specularStrength) * albedo / math.pi + specular * specularStrength)
    return ambient * albedo + direct * keyIntensity * normalLight


class Pytorch3dFaceRenderer:
    def __init__(self, geometry: dict, *, device: str = "cuda", imageSize: int | tuple[int, int] = 768,
                 frameGeometry: bool = True):
        imageShape = (imageSize, imageSize) if isinstance(imageSize, int) else tuple(imageSize)
        if len(imageShape) != 2 or min(imageShape) < 32:
            raise ValueError("imageSize must be at least 32")
        self.device, self.imageSize = device, imageSize
        self.imageShape, self.frameGeometry = imageShape, frameGeometry
        tensor = lambda values, dtype=torch.float32: torch.as_tensor(values, dtype=dtype, device=device)
        vertices = tensor(geometry["vertices"])
        self.center = (vertices.amax(0) + vertices.amin(0)) / 2
        self.scale = 1.9 / (vertices[:, :2].amax(0) - vertices[:, :2].amin(0)).max()
        if frameGeometry:
            vertices = (vertices - self.center) * self.scale
        self.faces = tensor(geometry["faces"], torch.int64)
        self.uvs = tensor(geometry["objUvs"])
        self.uvFaces = tensor(geometry["uvFaces"], torch.int64)
        self.mesh = Meshes(verts=[vertices], faces=[self.faces])
        self.cornerFrames = tensor(geometry["cornerTangentFrames"])
        if self.cornerFrames.shape != (len(self.faces), 3, 3, 3):
            raise ValueError("Expected (face, corner, XYZ, TBN-column) tangent frames")
        defaultValidity = torch.linalg.vector_norm(self.cornerFrames[..., 2], dim=-1).amin(-1) > 0.5
        self.faceTangentValid = (tensor(geometry["faceTangentValidMask"], torch.bool)
                                 if "faceTangentValidMask" in geometry else defaultValidity)
        self.rasterSettings = RasterizationSettings(image_size=imageSize, blur_radius=0,
                                                    faces_per_pixel=1, cull_backfaces=True)
        self.fragments = None
        self.normals = None

    @torch.no_grad()
    def setView(self, azimuth: float = 0, elevation: float = 0):
        rotation, translation = look_at_view_transform(dist=4, elev=elevation, azim=azimuth, device=self.device)
        cameras = FoVOrthographicCameras(device=self.device, R=rotation, T=translation,
                                         min_x=-1.10, max_x=1.10, min_y=-1.10, max_y=1.10,
                                         znear=0.1, zfar=10)
        self.rasterizeCamera(cameras, perspective=False)

    @torch.no_grad()
    def setPerspectiveCamera(self, intrinsics, rotation, translation):
        """Use OpenCV camera matrices for the exact, unframed input geometry.

        Intrinsics use integer-centered image pixels, as cv2.projectPoints.
        PyTorch3D screen coordinates use boundary coordinates with pixel centers
        at index+0.5. Only the raster camera receives that explicit half-pixel
        shift. ``projectionCamera`` retains direct OpenCV-screen equivalence.
        """
        if self.frameGeometry:
            raise ValueError("Photo cameras require frameGeometry=False; do not auto-center the mesh")
        from pytorch3d.utils import cameras_from_opencv_projection

        tensor = lambda values: torch.as_tensor(values, dtype=torch.float32, device=self.device)
        matrix = tensor(intrinsics).clone()
        if matrix.shape != (3, 3):
            raise ValueError("Expected one 3x3 intrinsic matrix")
        size = tensor([self.imageShape])
        cameraArgs = {"R": tensor(rotation)[None], "tvec": tensor(translation).reshape(1, 3), "image_size": size}
        self.projectionCamera = cameras_from_opencv_projection(camera_matrix=matrix[None], **cameraArgs)
        matrix[:2, 2] += 0.5
        rasterCamera = cameras_from_opencv_projection(camera_matrix=matrix[None], **cameraArgs)
        self.rasterizeCamera(rasterCamera, perspective=True)

    @torch.no_grad()
    def rasterizeCamera(self, cameras, *, perspective: bool):
        self.fragments = MeshRasterizer(cameras=cameras, raster_settings=self.rasterSettings)(self.mesh)
        self.visible = self.fragments.pix_to_face[0, ..., 0] >= 0
        pixelFaces = self.fragments.pix_to_face[0, ..., 0].clamp_min(0)
        self.validTangentPixels = self.visible & self.faceTangentValid[pixelFaces]
        interpolate = lambda values: interpolate_face_attributes(
            self.fragments.pix_to_face, self.fragments.bary_coords, values)[0, ..., 0, :]
        self.frames = interpolate(self.cornerFrames.reshape(-1, 3, 9)).reshape(*self.imageShape, 3, 3)
        self.positions = interpolate(self.mesh.verts_packed()[self.faces])
        cameraCenter = cameras.get_camera_center()[0]
        if perspective:
            self.viewDirections = functional.normalize(cameraCenter - self.positions, dim=-1)
        else:
            # Orthographic view rays are parallel, independent of surface position.
            self.viewDirections = functional.normalize(cameraCenter, dim=-1).expand_as(self.positions)
        # Invalid UV derivatives must not erase the underlying geometry normal.
        self.normals = functional.normalize(interpolate(self.mesh.verts_normals_packed()[self.faces]), dim=-1)

    @torch.no_grad()
    def sampleAtlas(self, values: np.ndarray) -> torch.Tensor:
        if self.fragments is None:
            raise RuntimeError("Call setView before sampling materials")
        values = np.asarray(values)
        if values.ndim == 2:
            values = values[..., None]
        maps = torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float32, device=self.device)[None]
        textures = TexturesUV(maps=maps, faces_uvs=[self.uvFaces], verts_uvs=[self.uvs], align_corners=True)
        return textures.sample_textures(self.fragments)[0, ..., 0, :]

    @torch.no_grad()
    def prepareMaterial(self, albedo: np.ndarray, coverage: np.ndarray, *,
                        specular: np.ndarray | None = None, tangentNormal: np.ndarray | None = None,
                        normalValidMask: np.ndarray | None = None) -> dict:
        # Only this diagnostic assumes native OpenDelight and WC use gamma 2.2.
        linearAlbedo = self.sampleAtlas(np.clip(albedo, 0, 1) ** 2.2)
        observed = self.sampleAtlas(coverage).squeeze(-1) >= 0.999
        # Missing observations stay visually identifiable, never skin-colored fill.
        linearAlbedo[~observed] = 0.07
        specularValues = (torch.full_like(linearAlbedo[..., :1], 0.22) if specular is None
                          else self.sampleAtlas(specular).clamp(0, 1))
        specularValues[~observed] = 0
        normals = self.normals
        if tangentNormal is not None:
            if normalValidMask is None:
                raise ValueError("A normal map requires an explicit valid mask")
            sampled = functional.normalize(self.sampleAtlas(tangentNormal), dim=-1)
            qualified = ((self.sampleAtlas(normalValidMask).squeeze(-1) >= 0.999)
                         & observed & self.validTangentPixels)
            detailNormals = applyTangentNormals(self.frames, sampled)
            normals = torch.where(qualified[..., None], detailNormals, normals)
        return {"albedo": linearAlbedo, "normals": normals, "specular": specularValues}

    @torch.no_grad()
    def render(self, material: dict, lightDirection=(-0.6, 0.5, 1), *, roughness=0.45,
               keyIntensity=2.5, ambient=0.18) -> np.ndarray:
        values = shadeGgx(material["albedo"], material["normals"], self.viewDirections,
                          material["specular"], lightDirection, roughness=roughness,
                          keyIntensity=keyIntensity, ambient=ambient)
        image = linearToSrgb(values).clamp(0, 1)
        image[~self.visible] = torch.tensor([0.105, 0.12, 0.14], device=self.device)
        if not torch.isfinite(image).all():
            raise FloatingPointError("Nonfinite rendered pixels")
        return image.cpu().numpy()
