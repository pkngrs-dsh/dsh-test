"""Prepare, without activating, an index-preserving client-skin Wrap bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faceAppearance.clientTemplate import readLegacyObj
from faceAppearance.correspondence import arrayHash


def fileHash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def writeObj(path, vertices, faces, arrayUvs, uvFaces, description):
    """Write standard V-up OBJ, retaining all vertex, face, and UV indices."""
    with Path(path).open("w") as stream:
        stream.write(f"# {description}\n# Standard OBJ UV: v up. No normals or material dependencies.\n")
        for vertex in vertices:
            stream.write("v " + " ".join(format(float(value), ".17g") for value in vertex) + "\n")
        for uv in arrayUvs:
            stream.write(f"vt {float(uv[0]):.17g} {float(1-uv[1]):.17g}\n")
        for face, uvFace in zip(faces, uvFaces):
            stream.write("f " + " ".join(f"{int(vertex)+1}/{int(uv)+1}" for vertex, uv in zip(face, uvFace)) + "\n")


def buildBundle(clientDir, outputDir):
    clientDir, outputDir = Path(clientDir), Path(outputDir)
    if outputDir.exists() and any(outputDir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty Wrap work directory: {outputDir}")
    manifestPath = clientDir / "manifest.json"
    client = json.loads(manifestPath.read_text())
    for name, expected in client["files"].items():
        if fileHash(clientDir / name) != expected:
            raise ValueError(f"Compiled source asset changed: {name}")
    arrays = np.load(clientDir / "template.npz", allow_pickle=False)
    sourcePath, fullPath = Path(client["sources"]["nicpMesh"]), Path(client["sources"]["fullMesh"])
    for key, path in (("nicpMesh", sourcePath), ("fullMesh", fullPath)):
        if fileHash(path) != client["templateHashes"][key]:
            raise ValueError(f"Original template identity changed: {key}")
    source, full = readLegacyObj(sourcePath), readLegacyObj(fullPath)
    matrixRow = np.eye(4)
    diagnostics = client["diagnostics"]
    matrixRow[:3, :3] = diagnostics["similarityScale"]*np.asarray(diagnostics["similarityRotationRowVector"])
    matrixRow[3, :3] = diagnostics["similarityTranslation"]
    inverseMatrixRow = np.linalg.inv(matrixRow)

    def transformed(vertices):
        return (np.c_[vertices, np.ones(len(vertices))] @ matrixRow)[:, :3]

    alignedSource, alignedFull = transformed(source["vertices"]), transformed(full["vertices"])
    fixedVertices, fixedFaces, fixedUvs = arrays["sourceVertices"], arrays["sourceFaces"], arrays["sourceUvs"]
    outputDir.mkdir(parents=True)
    (outputDir / "originals").mkdir()
    (outputDir / "selections").mkdir()
    shutil.copyfile(sourcePath, outputDir / "originals/female260429NicpCentimeters.obj")
    shutil.copyfile(fullPath, outputDir / "originals/female260429FullCentimeters.obj")
    sourceFile, targetFile, referenceFile = "clientSkinFloating.obj", "uvFaceFusionFixed.obj", "clientFullAlignedReference.obj"
    writeObj(outputDir / sourceFile, alignedSource, source["faces"], source["uvs"], source["uvFaces"],
             "Floating: old 2795-vertex client skin; template centimeters transformed to UFF canonical")
    writeObj(outputDir / targetFile, fixedVertices, fixedFaces, fixedUvs, fixedFaces,
             "Fixed: unchanged UFF canonical geometry; original native V-down UV converted to standard OBJ V-up")
    writeObj(outputDir / referenceFile, alignedFull, full["faces"], full["uvs"], full["uvFaces"],
             "Reference only: original full 3544-vertex client template in the same initial UFF space; do not wrap accessories")
    weights, static = arrays["geometryWeights"], arrays["staticMask"]
    qualifiedVertices = np.flatnonzero((weights > 0) & ~static)
    qualifiedFaces = np.flatnonzero(((weights > 0) & ~static)[source["faces"]].all(1))
    selections = {"qualifiedSkinVertexIndices.json": qualifiedVertices,
                  "qualifiedSkinFaceIndices.json": qualifiedFaces,
                  "staticVertexIndices.json": np.flatnonzero(static),
                  "staticRegionFaceIndices.json": np.asarray(json.loads(Path(client["sources"]["staticFaceMask"]).read_text()), np.int64),
                  "unqualifiedSkinFaceIndices.json": np.setdiff1d(np.arange(len(source["faces"])), qualifiedFaces)}
    for name, indices in selections.items():
        (outputDir / "selections" / name).write_text(json.dumps(indices.tolist())+"\n")
    np.savez_compressed(outputDir / "selectionData.npz", geometryWeights=weights, staticMask=static,
                        nicpToFullIds=arrays["nicpToFullIds"], qualifiedSkinVertexIndices=qualifiedVertices,
                        qualifiedSkinFaceIndices=qualifiedFaces)
    loadedSource, loadedTarget, loadedReference = [readLegacyObj(outputDir / name) for name in (sourceFile, targetFile, referenceFile)]
    for loaded, original in ((loadedSource, source), (loadedReference, full)):
        for key in ("faces", "uvFaces"):
            if not np.array_equal(loaded[key], original[key]):
                raise ValueError(f"Export reordered {key}")
        if not np.allclose(loaded["uvs"], original["uvs"], rtol=0, atol=1e-15):
            raise ValueError("Export changed original source UV values")
    if not np.array_equal(loadedTarget["vertices"], fixedVertices) or not np.array_equal(loadedTarget["faces"], fixedFaces):
        raise ValueError("Fixed target geometry changed")
    if not np.array_equal(loadedTarget["uvs"].astype(np.float32), fixedUvs.astype(np.float32)):
        raise ValueError("Target standard OBJ UV conversion failed")
    restored = (np.c_[loadedSource["vertices"], np.ones(len(alignedSource))] @ inverseMatrixRow)[:, :3]
    roundTripError = float(np.max(np.abs(restored-source["vertices"])))
    review = {"status": "pending", "resultFile": "clientSkinRegisteredToUff.obj", "topologyChecked": False,
              "eyeCornersAndEyelidsChecked": False, "lipContactAndMouthCornersChecked": False,
              "noseAlarAndNostrilBoundariesChecked": False, "jawAndChinChecked": False,
              "unmatchedBackHeadEarsNeckPreserved": False, "foldsAndSelfIntersectionsChecked": False,
              "notes": "未检查的项目请保持 false；结构一致不等于解剖配准正确。"}
    (outputDir / "registrationReview.template.json").write_text(json.dumps(review, ensure_ascii=False, indent=2)+"\n")
    readme = """# 自有数字人皮肤拓扑 → UFF：Wrap 工作包 v1

本包仅准备离线配准，**尚未启用到流水线，也不是已经通过审核的对应资产**。目标是把照片重建结果可靠地转回已有数字人的自有拓扑，保持消费端现有顶点绑定。

## 需要加载的文件

| 角色 | 文件 | 顶点 / 三角面 / UV |
| --- | --- | --- |
| 固定目标，保持坐标不动 | `uvFaceFusionFixed.obj` | 6986 / 13632 / 6986 |
| 可形变源，只配准这个皮肤网格 | `clientSkinFloating.obj` | 2795 / 5532 / 2953 |
| 只供观察附件位置，不参与配准 | `clientFullAlignedReference.obj` | 3544 / 6732 / 3733 |

源是旧工程 **female/260429** 的 NICP 皮肤子网格，已用当前编译器记录的相似变换粗对齐至 UFF canonical 空间。它不是新建的 FLAME 网格。固定目标仍是原始 UFF canonical 几何。这个空间的数值单位没有认证为米或厘米。

三个主 OBJ 都采用标准 UV：U 向右、V 向上。目标从 UFF 原生数组 V-down 明确转换为 `1-v`；它与早先 `assets/wrapPreparation/uvFaceFusionFixed.obj` 的 UV 写法不同，但目标顶点及面索引完全相同。请使用本包内这对输入，不混用旧包。原源文件逐字节保存在 `originals/`，其中全模含已有眼部、口腔、眉毛和睫毛；原始坐标单位是厘米。原文件的历史 MTL 引用并不完整，本工作包按灰模几何查看即可。

## 配准方式

1. 固定目标使用 `uvFaceFusionFixed.obj`，可形变源使用 `clientSkinFloating.obj`。只移动和形变源，不移动目标。当前粗对齐只是起点，不保证眼角、唇线等解剖位置已对齐。
2. 用明确的解剖对应补充约束：左右内外眼角、上/下眼睑、鼻翼和鼻孔边界、上下唇边缘、嘴角、唇接触线、下颌和下巴。上下眼睑、上下唇不能互换，也不要跨开口连接。检查正面及两侧深度，不能只检查正面投影。
3. 后脑、耳、颈部及任一方没有共同表面的区域，目前保留原型形状，不要强迫吸附至 UFF。鼻孔深处、口腔内部和眼窝内部也不应用外表面最近点强贴。以自由区域、排除贴合或保持约束处理，并保留这些面的原始索引。
4. **眼球、牙齿、舌和眉睫附件不参加这次 Wrap。** 只提交2795顶点的源皮肤结果；完整3544顶点模型仅用于观察。后续流程按原索引把合格皮肤形变接回完整模型并处理附件。
5. 全程保留源顶点、面及 UV 的数量和顺序；不要焊点、删面、细分、重拓扑、重新三角化或重展 UV。不把参考完整模型合并进源。
6. 将形变最终落实到源的顶点位置，导出 `clientSkinRegisteredToUff.obj`。导出坐标仍在固定 UFF 空间，**不要恢复成厘米，也不要乘0.01**。输出保留2795顶点、5532面、2953 UV及原始索引。重新导入确认位置，避免变换仅存于软件节点而没有进入导出顶点。
7. 保存 `.wrap` 工程以及点对、排除区域选择。配准结果需再检查三角面翻折、自交和局部解剖边界；接近目标表面本身不能证明正确。可填写 `registrationReview.template.json`，未检查项保持 false。

## 辅助选择

`selections/` 内是**零基索引 JSON 列表**，索引针对2795顶点/5532面的源皮肤，不是完整3544模板，也不是 OBJ 文本的一基编号。这些文件是通用审计/脚本交换数据，未声明为某版 Wrap 的原生选择文件，可按版本导入或参照建立选择。

- `qualifiedSkinVertexIndices.json` 与 `qualifiedSkinFaceIndices.json`：当前候选 UV 映射通过距离/法向/接缝门控的区域，仅供参考，不能代替人工解剖审核。
- `staticVertexIndices.json` 与 `staticRegionFaceIndices.json`：旧非面部（含耳）掩码，当前原型保持的区域。538个静态顶点全部保留；不要删除它们。
- `unqualifiedSkinFaceIndices.json`：其余未取得候选映射资格的面。未通过资格不等于需要删除，也不等于都应该自由移动。
- `selectionData.npz`：相同选择及原有连续权重、NICP→full顶点索引，仅供程序读取。

这些选择不会裁剪或改变源拓扑，也没有作为强制 Wrap 约束写入模型。

## 坐标与归档

`manifest.json` 的两个4×4矩阵采用**行向量右乘**：`[x,y,z,1] @ matrixRow`。

- `templateCentimetersToUffInitialMatrixRow`：原始厘米模板 → 本包粗对齐 UFF 空间。
- `uffInitialToTemplateCentimetersMatrixRow`：上述变换的精确逆，用于程序恢复模板坐标；请不要在导出配准结果前应用。

逆矩阵仅撤销初始整体相似变换，不会撤销 Wrap 的局部形变。它不能独自证明最终绑定、碰撞或表情变形正确。目录中的源/目标哈希和输入校验用于检测误换版本，不能证明配准已完成。

最终希望收到 `clientSkinRegisteredToUff.obj`、`.wrap`、对应点及排除选择，最好附正面、左右侧面和眼口鼻近景。无需另行提交 UFF 变形到自有模板的网格。审核通过后才编译和启用新资产，本包不会自动替换现有候选映射。
"""
    (outputDir / "README.md").write_text(readme, encoding="utf-8")
    validation = {"preparationValid": True, "registrationAccepted": False, "semanticReviewRequired": True,
                  "sourceVertices": len(source["vertices"]), "sourceFaces": len(source["faces"]),
                  "sourceUvs": len(source["uvs"]), "targetVertices": len(fixedVertices),
                  "targetFaces": len(fixedFaces), "sourceIndicesPreserved": True, "sourceUvsPreserved": True,
                  "targetGeometryUnchanged": True, "targetNativeUvRoundTripExactFloat32": True,
                  "originalTemplateRoundTripMaxAbsoluteErrorCentimeters": roundTripError,
                  "qualifiedSkinVertices": len(qualifiedVertices), "qualifiedSkinFaces": len(qualifiedFaces),
                  "staticVertices": int(static.sum())}
    (outputDir / "inputValidation.json").write_text(json.dumps(validation, indent=2)+"\n")
    result = {"schemaVersion": 1, "status": "preparedNotActivated", "bundleId": "legacyFemaleClientWrapV1",
              "fixedTarget": targetFile, "floatingSource": sourceFile, "fullReference": referenceFile,
              "expectedResult": "clientSkinRegisteredToUff.obj", "sourceTemplate": "female/260429 NICP skin",
              "resultCoordinateFrame": "uvFaceFusionCanonical", "resultMetricUnits": "uncalibrated",
              "objUvOrigin": "vUp", "internalArrayUvOrigin": "vDown",
              "matrixConvention": "homogeneous row vector [x,y,z,1] @ matrixRow",
              "templateCentimetersToUffInitialMatrixRow": matrixRow.tolist(),
              "uffInitialToTemplateCentimetersMatrixRow": inverseMatrixRow.tolist(),
              "clientBundleManifestSha256": fileHash(manifestPath),
              "clientBundleTemplateSha256": fileHash(clientDir / "template.npz"),
              "sourceOriginalObjSha256": fileHash(sourcePath), "fullOriginalObjSha256": fileHash(fullPath),
              "sourceAlignedObjSha256": fileHash(outputDir / sourceFile), "targetStandardObjSha256": fileHash(outputDir / targetFile),
              "sourceTopologyHash": arrayHash(source["faces"].astype(np.int64), source["uvs"].astype(np.float64), source["uvFaces"].astype(np.int64)),
              "targetNativeTopologyHash": arrayHash(fixedFaces.astype(np.int64), fixedUvs.astype(np.float32)),
              "scope": "Only wrap existing source skin; original full topology and accessories are reference only",
              "activation": "None; separate anatomical review and compilation required", "inputValidation": validation}
    result["files"] = {str(path.relative_to(outputDir)): fileHash(path) for path in sorted(outputDir.rglob("*")) if path.is_file()}
    (outputDir / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clientDir", type=Path, default=root / "assets/clientTemplates/legacyFemaleV1")
    parser.add_argument("--outputDir", type=Path, default=root / "assets/clientWrapPreparation/v1")
    args = parser.parse_args()
    result = buildBundle(args.clientDir, args.outputDir)
    print(json.dumps({"outputDir": str(args.outputDir), **result["inputValidation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
