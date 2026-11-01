#!/usr/bin/env bash
set -euo pipefail

releaseRoot="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
runtimePrefix="${FACE_APPEARANCE_CONDA_PREFIX:-${CONDA_PREFIX:-$releaseRoot/.venv}}"
pythonExecutable="${FACE_APPEARANCE_PYTHON:-$runtimePrefix/bin/python}"
configPath="${FACE_APPEARANCE_CONFIG:-$releaseRoot/configs/stableLegacyService.json}"

if [[ ! -x "$pythonExecutable" ]]; then
  printf 'Python executable not found; set FACE_APPEARANCE_PYTHON: %s\n' "$pythonExecutable" >&2
  exit 1
fi

export PYTHONUTF8=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$releaseRoot/src"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
export PATH="$runtimePrefix/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-2}"
export FACE_APPEARANCE_COLMAP="${FACE_APPEARANCE_COLMAP:-$releaseRoot/.runtimes/colmap/bin/colmap}"

cd "$releaseRoot"
exec "$pythonExecutable" -X utf8 -m faceAppearance.cli \
  --config "$configPath" serve-legacy \
  --host "${FACE_APPEARANCE_HOST:-127.0.0.1}" \
  --port "${FACE_APPEARANCE_PORT:-7860}" \
  --outputRoot "${FACE_APPEARANCE_TASK_ROOT:-$releaseRoot/runs/legacyService}" \
  --queueCapacity "${FACE_APPEARANCE_QUEUE_CAPACITY:-2}" \
  --maleAssemblyAsset "${FACE_APPEARANCE_MALE_ASSET:-assets/clientTemplates/legacyMaleDirectServiceV1}" \
  --femaleAssemblyAsset "${FACE_APPEARANCE_FEMALE_ASSET:-assets/clientTemplates/legacyFemaleV1}"
