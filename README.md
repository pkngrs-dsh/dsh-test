# dsh-test

Test repository for dsh-superagent.

## 项目简介

dsh-test 是一个用于 dsh-superagent 的测试仓库。

## 项目地图

仓库结构、重建链路、服务入口和模块职责见 [项目地图](docs/projectMap.md)。

## FaceAppearance 主体代码

此仓库收录 FaceAppearance 多视角人脸几何与外观重建原型的主体代码。目标是从 4–16 张照片重建可重打光人脸，并生成保留旧客户端顶点编号的装配资产。

| 目录 | 用途 |
|---|---|
| `src/faceAppearance/` | 模型适配、几何与贴图处理、装配、服务和网页预览 |
| `tests/` | 单元测试、服务契约测试及资产相关集成测试 |
| `configs/` | 重建、客户端装配和服务配置 |
| `tools/` | 资产准备、诊断、渲染及研究实验脚本 |
| `docs/` | 项目地图、源码运行与外部依赖说明 |
| `deploy/` | 可修改的部署模板与依赖清单 |

模型权重、照片、客户端模板、配准资产、实验结果、本地环境，以及六个完整上游项目均不随此仓库提交。代码包含可选实验模块，它们不一定被当前服务配置启用。完整推理需要按 [源码与部署说明](docs/sourceSetup.md) 配置外部依赖和资产。

### Python 入口

```bash
# 使用合适的 Python 环境安装主体包；GPU 依赖另行配置。
python -m pip install -e .
face-appearance --help
face-appearance-client --help

# 准备好模型、上游依赖与资产后运行重建。
face-appearance --config configs/preview512.json reconstruct \
  --input /path/to/photos --output runs/example
```

### 不依赖模型资产的基础测试

```bash
python -m pip install pytest httpx
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/testFaceSelection.py tests/testFaceCrop.py \
  tests/testImageColor.py tests/testCameraData.py tests/testTexture.py
```

部分集成测试与历史实验工具需要本地模板、原始照片或已保存的运行结果。未提供这些数据时，完整测试目录不保证通过；测试数据和环境要求见 [源码与部署说明](docs/sourceSetup.md)。

上游代码的来源和许可证保留在原目录，索引见 [第三方代码说明](THIRD_PARTY_NOTICES.md)。
