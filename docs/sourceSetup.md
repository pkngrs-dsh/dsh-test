# 主体源码的安装、测试与部署

本仓库提交 `src`、`tests`、`configs`、`tools`、依赖声明和通用部署模板。模型权重、照片、客户端模板、配准数据、运行结果、实验报告、完整上游工程和本机 Python 环境单独管理。克隆后可以阅读源码、运行不依赖外部资产的测试；真实重建需要先补齐下述环境及资产。

`tools` 同时包含通用工具与历史实验入口。部分脚本依赖特定的 `runs`、`reports` 或本地数据集，不能将所有工具视为开箱即用的服务步骤。配置中的功能开关决定实际处理链，实验模块的存在不代表默认启用。

## 基础安装与 CPU 测试

从仓库根目录创建独立环境：

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install pytest==8.3.4 httpx==0.28.1
.venv/bin/python -m faceAppearance.cli --help
```

以下测试使用合成输入或仓库内小型测试素材，适合没有权重、模板、历史实验目录的源码副本：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q \
  tests/testFaceSelection.py tests/testFaceCrop.py tests/testImageColor.py \
  tests/testCameraData.py tests/testTexture.py tests/testPhotoLighting.py \
  tests/testShDelight.py tests/testBarycentricDeformation.py \
  tests/testFixedBoundaryDeformation.py tests/testFrequencyFusion.py \
  tests/testObservedSampling.py tests/testSurfaceFairing.py \
  tests/testMeshShapeConstraints.py tests/testAtlasTransfer.py
```

完整测试集还包含真实模板、上游模型和历史实验验证，直接运行全部测试可能因这些外部文件缺失而失败。CPU 单元测试通过不代表 GPU 重建效果、性能或客户端表情绑定已经通过验收。

## GPU 环境

原型采用 Python 3.10、PyTorch 2.3.1 / CUDA 12.1、torchvision 0.18.1，渲染路径使用 PyTorch3D 0.7.8 与 nvdiffrast 0.4.0。可使用 [Conda 基础环境](../deploy/environment.yml) 代替上述 venv：

```bash
conda env create -f deploy/environment.yml
conda activate faceappearance
python -m pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e . -r requirements.runtime.txt
```

PyTorch3D 和 nvdiffrast 需从各自上游提供的兼容构建或源码单独安装；本仓库没有附带编译包。构建 CUDA 扩展需要 CUDA toolkit 和兼容 C++ 编译器，只有显卡驱动不足以完成编译。`CUDA_HOME` 应指向本机实际 toolkit，首次 JIT 编译时间与推理时间分别计算。

[requirements.lock.txt](../deploy/requirements.lock.txt) 保留原 Linux x86_64 / Python 3.10 服务依赖快照；它不是包含 wheel 哈希的完整锁文件，也未保证在新机器上的依赖解析和安装成功。迁移时应记录实际解析版本及安装日志。该服务快照不含 PyTorch3D、WildCap 全部依赖或测试依赖；`requirements.runtime.txt` 提供原型增量依赖。

男性 `directColmap` 路径要求支持 `mesh_texturer` 的 COLMAP；单纯安装任意同名版本并不充分。[colmap-explicit.txt](../deploy/colmap-explicit.txt) 是原 linux-64 独立环境快照，可用 `conda create --prefix /path/to/colmap-env --file deploy/colmap-explicit.txt` 单独恢复，再执行 `colmap-env/bin/colmap mesh_texturer --help` 检查。它不应与 Python 推理环境合并。用 `FACE_APPEARANCE_COLMAP` 指定可执行文件的绝对路径，或使用配置中相对于项目根目录的路径。

## 外部源码、权重与模板

运行时按固定目录契约查找文件。以下路径相对于仓库根目录，均需按相应来源与使用协议另行准备：

| 路径 | 用途 |
|---|---|
| `01_UVFaceFusion/uvfacefusion/` | UFF 上游源码、VGGT 依赖，以及 `assets/canonical_mesh_flame.ply`、`canonical_mesh_flame_uvs.npy` |
| `01_UVFaceFusion/weights/` | `vggt_weights.pt`、`uv.ckpt`、`fusion_net.pth` 和 `facer_hub_checkpoints` 检测／分割权重 |
| `02_OpenDelight/opendelight/` | OpenDelight 网络源码及 `config/delight_base_network.yaml` |
| `02_OpenDelight/weights/opendelight/` | `base_delight_network.pth`；启用增强时还需 `unet_enhancer.pth` |
| `thirdParty/facer/`、`thirdParty/pixel3dmm/`、`thirdParty/ibug/` | 分割、UV／法线先验、人脸检测与对齐的本地源码依赖 |
| `assets/models/ibug/` | `Resnet50_Final.pth`、`2dfan2_alt.pth` |
| `assets/clientTemplates/` | 带 manifest、数组及材质的客户端装配模板；必须与消费端顶点编号和绑定匹配 |
| `assets/clientLandmarks/legacyMediapipe468/` | 旧客户端语义重心绑定；对应 JSON 文件与模板拓扑共同校验 |
| `04_STFR/weights/registration/` | 启用眼睑检测／男性直接配准时使用的 `mergefaneye_d2_ep40_sym_DM_ce_gray_noblend.onnx` |
| `03_WildCap/wildcap/`、`03_WildCap/weights/`、`03_WildCap/data/` | 启用 WildCap 后使用的网络、渲染器、权重及 ICT 固定数据 |
| `assets/registration/wrapV1/` | 启用规范空间传递时使用的已编译配准数据 |
| `assets/weights/pixel3dmm/normals.ckpt` | 可选法线先验权重 |

适配器启动时加载本地模型，不会在请求过程中自动下载缺失权重。`configureRuntime` 默认使用项目内缓存并关闭 Hugging Face 在线下载，已有调用方环境变量优先。详细缺失文件由适配器的启动检查报告。

少量提取的上游代码随 `src/faceAppearance/vendor` 提交；它们的 `LICENSE` 与 `SOURCES.md` 保留在对应目录。Three.js 的许可证和来源保留在 `src/faceAppearance/web/vendor`。这些文件不会替代未随源码提交的模型、网格和其他资产的使用协议。

## 本地运行

资产和环境就绪后，在仓库根目录使用新输出目录：

```bash
python -m faceAppearance.cli --config configs/preview512.json reconstruct \
  --input /path/to/photos --output runs/example

python -m faceAppearance.cli --config configs/clientService.json serve-client \
  --host 127.0.0.1 --port 8000 --outputRoot runs/clientService
```

输入目录应直接包含同一人的 4–16 张不同角度照片。`serve-client` 提供 multipart `/v1/jobs` 协议及 `/viewer/` 预览；旧客户端 JSON/URL 协议使用独立的 `serve-legacy`。每块 GPU 使用一个服务进程，作业串行执行；输出目录应独立于代码版本。

`configs/localWarmLegacyService.json` 的照片路径是示例，预热默认关闭。准备自己的四张预热图片、修改路径并确认模板性别后，才将 `warmupEnabled` 设为 `true`。模板、权重和测试照片均留在本地，不提交到 Git。

## 旧客户端服务部署模板

[run-legacy.sh](../deploy/run-legacy.sh) 使用配置文件启动 `serve-legacy`。默认配置是 `configs/stableLegacyService.json`，包含男性直接配准和渲染下颌修正；它要求上述客户端模板、检测模型与 COLMAP 都已就绪。模板只演示部署方式，不代表服务已启动。

脚本接受 `FACE_APPEARANCE_PYTHON` 指定 Python 可执行文件；也可用 `FACE_APPEARANCE_CONDA_PREFIX` 指定环境根目录，或使用当前已激活的 Conda 环境／项目 `.venv`。`FACE_APPEARANCE_CONFIG`、`FACE_APPEARANCE_COLMAP`、`FACE_APPEARANCE_TASK_ROOT` 和男女模板路径可单独覆盖。

```bash
# 在已配置环境及资产的项目目录启动本机旧协议服务。
export FACE_APPEARANCE_PYTHON=/path/to/env/bin/python
export FACE_APPEARANCE_COLMAP=/path/to/colmap-env/bin/colmap
export FACE_APPEARANCE_PUBLIC_BASE_URL=http://127.0.0.1:7860
bash deploy/run-legacy.sh
```

`PUBLIC_BASE_URL` 是客户端可访问的结果下载 origin；远程客户端须替换成本机环回以外的实际地址。OSS 模式需完整设置相应 `FACEV2_OSS_*` 环境变量。示例中没有真实账号、凭据或生产域名。

[faceappearance.service](../deploy/faceappearance.service) 是用户级 systemd 模板，示例布局为 `/opt/faceappearance/current`（源码）、`/opt/faceappearance/shared`（环境文件和任务）。部署前由管理员准备目录和所有权，并修改 unit 与 [service.env.example](../deploy/service.env.example) 中的路径；将环境示例复制到仓库外 `shared/service.env`，权限设为 `0600`。确认命令行启动成功后再把 unit 放入 `~/.config/systemd/user/`，执行 `systemctl --user daemon-reload` 和 `systemctl --user enable --now faceappearance.service`。脚本本身不读取 `.env`，由 systemd 的 `EnvironmentFile` 注入环境。

`tools/legacyPreviewRemote.py` 是可选 SSH 联调助手。发布代码已将远端 URL、SSH 主机、Python 和助手路径改为环境配置；远端进程还需设置 `FACE_APPEARANCE_PREVIEW_SHARED_ROOT`、`FACE_APPEARANCE_PREVIEW_UPLOAD_PREFIX`、`FACE_APPEARANCE_PREVIEW_PUBLIC_URL`。本仓库没有预置可连接的生产主机。
