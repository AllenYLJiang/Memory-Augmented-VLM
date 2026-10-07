#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
python -m pip install -r requirements.txt
python -c 'import torch; from transformers import Qwen3VLForConditionalGeneration; assert torch.cuda.is_available(), "CUDA required"; print(torch.__version__, torch.cuda.get_device_name(0))'
