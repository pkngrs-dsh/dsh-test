# WildCap adapter provenance

The implementation in `../../adapters/wildcap.py` adapts the inference equations
from the checked-in `03_WildCap/wildcap/run_wildcap.py`, authored by Yuxuan Han et
al., project https://github.com/yxuhan/WildCap (GPL version 3; license alongside
this file). The original repository is left unchanged.

`backend.py` executes the original network and renderer definitions directly
from that local source tree. It creates private package names, skips unused
evaluation/transformer imports and rewrites only three absolute imports in
`utils/lighting.py`. It does not run `run_wildcap.py`, its argument parser, or
its sampler. No arbitrary model source is downloaded or imported.

Retained conventions: seven diffuse/normal/specular channels; gamma 2.2 diffuse;
UV position encoding using `arange(resolution)/resolution`; fixed ICT atlas and
1024 reference crop `(x1,x2,y1,y2)=(128,896,64,832)`; Karras 1000-step schedule;
noise-initialized prior, eta=1, 20-candidate DPPS; Split-Sum plus local lighting,
upstream masks/specular/normal initialization, and skin-mean normalization.

Explicit adaptations:

- Models and fixed assets load once. Lights, optimizers, noise, geometry and
  masks are created for each request; request RNG state is restored afterwards.
- `iterations` is the actual number of updates, taking `iterations+1` sigmas
  from the tail of the 1000-step schedule. Upstream's default 600 sigma entries
  executes 599 updates; pass 599 when comparing that exact setting.
- Input visibility masks exclude missing observations from all likelihoods.
  Guidance blur is normalized by available support to avoid zero-filled holes
  leaking into observed pixels. Artifact masks remain a separate lighting mask.
- Skin tone may be supplied or measured from observed, non-artifact pixels
  inside the upstream skin mask. Generated texels are explicitly marked.
- DPPS candidates are streamed to limit peak memory. This preserves the
  candidate-selection objective, but random draws need not be bit-identical to
  upstream's batched random generation on CUDA.
- Outputs retain the chosen working resolution, normalize tangent vectors,
  clamp encoded image values for serialization, and report clipping. No 4K
  superresolution or full-head completion is asserted.
- Render-plugin JIT preparation is serialized and temporarily uses explicit
  UTF-8 for Torch 2.3's build-cache text helper. This supports non-ASCII project
  paths after native vision inference changes the C locale; the helper is
  restored afterwards and no renderer equations are changed.

The local `splitsum` files include separate NVIDIA notices and CUDA source
dependencies; they remain in their original tree. Upstream's resizer credits
Assaf Shocher, and its position embedding credits NeRF. This adapter does not
change the licensing of the source, weights, ICT mesh or other assets.
