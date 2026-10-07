@echo off
setlocal
cd /d "%~dp0"
if not defined TORCH_INDEX_URL set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
python -m pip install torch torchvision --index-url %TORCH_INDEX_URL%
if errorlevel 1 exit /b 1
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python -c "import torch; from transformers import Qwen3VLForConditionalGeneration; print('torch',torch.__version__,'CUDA',torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA driver or PyTorch build unavailable'; print(torch.cuda.get_device_name(0))"
exit /b %errorlevel%
