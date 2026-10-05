#!/usr/bin/env bash
# Rebuild only FlashAttention against the active training environment's torch ABI.
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-python3}
FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION:-2.8.3}
export MAX_JOBS=${MAX_JOBS:-8}
export NVCC_THREADS=${NVCC_THREADS:-2}
export FLASH_ATTENTION_FORCE_BUILD=TRUE

"$PYTHON_BIN" - <<'PY'
import os
import torch
from torch.utils.cpp_extension import CUDA_HOME

print('Building for torch', torch.__version__, 'CUDA', torch.version.cuda,
      'CXX11_ABI', torch._C._GLIBCXX_USE_CXX11_ABI, flush=True)
if not CUDA_HOME or not os.path.isfile(os.path.join(CUDA_HOME, 'bin', 'nvcc')):
    raise SystemExit('Set CUDA_HOME to an installed CUDA toolkit containing bin/nvcc.')
print('CUDA toolkit:', CUDA_HOME, flush=True)
PY

# No dependency upgrades; build isolation would otherwise select a different torch.
uv --no-cache pip install --python "$PYTHON_BIN" --no-deps --no-build-isolation \
  --no-binary flash-attn --reinstall-package flash-attn "flash-attn==$FLASH_ATTN_VERSION"
"$PYTHON_BIN" - <<'PY'
import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
print('FlashAttention imports successfully against torch', torch.__version__)
PY
