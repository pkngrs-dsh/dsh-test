# UVFaceFusion vendored computation

Source: https://github.com/grignarder/UVFaceFusion, local revision `3de1c9a`.
License: GPL-3.0, retained in `LICENSE`.

- `models.py`: upstream network definitions, unchanged except provenance header. Original attribute names must remain compatible with the released state dictionary.
- `meshSampling.py`: upstream `reconstruct_mesh_utils.py`, unchanged except provenance header.
- `inference.py`: four pure functions extracted from upstream `infer.py`; the command-line parser, model construction, asset loads and execution loop are excluded.
- `utilities.py`: only Umeyama alignment, UV rasterization and masked Gaussian blur extracted from upstream `utils.py`; the nvdiffrast import is lazy.

The old FaceReconstruction/v2 extraction and registry informed this separation, but no file from v2 is imported or required at runtime. Upstream snake_case names are retained only inside these vendored files. New adapter code uses camelCase.
