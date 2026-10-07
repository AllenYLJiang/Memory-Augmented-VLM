# V919 Local Qwen 8B, Portable Comparison Edition


## 1. Install on the Other Computer

Unzip to a writable directory, e.g. `D:\experiments\v919_local_qwen8b_portable`.
The ZIP does **not** contain model weights, videos, previous API responses, or keys.

Windows Anaconda Prompt:

```bat
conda create -n v919_qwen8b python=3.11 -y
conda activate v919_qwen8b
cd /d D:\experiments\v919_local_qwen8b_portable
install_windows.cmd
conda install -c conda-forge ffmpeg -y
```

Linux/WSL with a working NVIDIA driver:

```bash
conda create -n v919_qwen8b python=3.11 -y
conda activate v919_qwen8b
bash install_linux.sh
conda install -c conda-forge ffmpeg -y
```

The installer defaults to the CUDA 12.8 PyTorch wheel index. Check the destination
driver with `nvidia-smi`; use the appropriate index from
[PyTorch's official installer](https://pytorch.org/get-started/locally/) if necessary.
Set `TORCH_INDEX_URL` before the install script to change it. No CPU fallback is allowed.
Default inference uses BF16, one GPU, SDPA, one generation at a time. Flash Attention 2
is optional on compatible systems, not required on Windows. A100 80 GB should have
substantial headroom for this configuration, but peak use must be measured locally.

Download models before entering offline execution (or copy complete local directories):

```bat
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir G:\Qwen3-VL-8B
hf download Qwen/Qwen3-8B --local-dir G:\Qwen3-8B
```

The second download is optional for V919. Official implementations:
[VLM model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
[LLM model card](https://huggingface.co/Qwen/Qwen3-8B).
We use the native Qwen3-VL model/processor and its chat template. The optional LLM
supports `enable_thinking`; thinking mode uses sampling rather than greedy decoding.

## 2. Freeze the Identical Pilot on the Destination Machine

Make the original training videos available on the new machine. Preserve video
filenames. This step searches for the **same fixed IDs**, not a new random sample.
It checks size and source hashes where the API media receipts exist; all destination
video hashes are frozen. Full model-file hashing can take a few minutes once per run start.

```bat
python -B run_local.py --stage 1 --tag v919_local_qwen8b_seed20260917 --train-root G:\Dataset\XDViolence\train --vlm-dir G:\Qwen3-VL-8B
```

On Linux replace roots, e.g. `--train-root /data/XDViolence/train --vlm-dir /models/Qwen3-VL-8B`.
Use `--device cuda:1` at Stage 1 to choose another GPU; otherwise `cuda:0`.
Do not change the model files, package source, precision, or dependencies mid-run.
The protocol includes their hashes/versions; changed settings need a fresh TAG.

**Recommended exact-frame transfer:** copy the original API run's `media` directory
into the newly prepared local run's `media` directory. Its images are not VLM outputs
and can be shared between model arms. The original directory is:

```text
ot_graph_advantage_conditional_ot_v3_revision/runs/governed_v919_effectiveness_20260917_r1/media
```

Example, after copying this directory to `D:\transfer\api_media`:

```bat
robocopy D:\transfer\api_media runs\v919_local_qwen8b_seed20260917\media /E
```

Robocopy exit codes 0..7 are normally nonfatal; inspect its summary. Do **not** copy
`cache`, `pilot/results`, `models`, or `terminal_refusals` from the API run.

Original JPEG hashes are checked before local VLM inference. Without transferring
images, the runner decodes the same frame indices and requires matching JPEG hashes
where known. Different decoder/JPEG versions may cause a mismatch: transfer the
original media rather than bypassing the check. Videos that had no API media receipt
are explicitly recorded as not yet pixel-paired with the API arm.

## 3. Collect, Fit, Calibrate and Validate Locally

Recommended first GPU check on the same frozen run (40 generations maximum,
intentionally pauses at its cumulative compute cap; not a smaller evaluation cohort):

```bat
python -B run_local.py --stage 2 --action run --tag v919_local_qwen8b_seed20260917 --approve-local-compute --approved-by researcher --max-attempts 40 --max-reserved-output-tokens 327680
```

Inspect output validity and `pilot/cost/attempts` for latency/memory, then resume
with the full reviewed local budget below. Existing responses remain cached.

```bat
python -B run_local.py --stage 2 --action run --tag v919_local_qwen8b_seed20260917 --approve-local-compute --approved-by researcher --max-attempts 14800 --max-reserved-output-tokens 121241600
```

The caps limit local work, not monetary billing. They are cumulative within this
phase, not reset on restart. Same command resumes completed requests/windows from
this **local-model-only** cache. A local OOM/device failure stops execution; it is not
automatically reclassified as a transport failure. Review the failure before any retry.
After an abrupt process/power interruption, an `in_flight` receipt without a saved
response also requires explicit investigation; do not delete the receipt or stale
operation lock without verifying that the old process has exited. This package
does not automatically authorize uncertain GPU attempts or replay failed generations.
Invalid JSON/schema/semantic outputs are retained with masks, never retried until
they look favorable. Do not shorten tokens or frames in-place to make a failed case pass.

Progress and offline re-evaluation:

```bat
python -B run_local.py --stage 2 --action status --tag v919_local_qwen8b_seed20260917
python -B run_local.py --stage 2 --action evaluate --tag v919_local_qwen8b_seed20260917
```

`evaluate` refuses partial-cohort fitting. Main files:

- `pilot/results/*.json`: six node/graph settings, C1/C2, all errors and masks.
- `cache/*.json`: exact prompts, parent hashes, image hashes, local raw responses.
- `pilot/cost/attempts/*.json`: tokens, latency, GPU allocated/reserved peaks.
- `pilot/feature_coverage.json`: validity and fallback coverage.
- `models/frozen.json`: certified ridge-logistic scorers and frozen thresholds.
- `pilot/fit_calibration_metrics.json`: non-held-out diagnostic metrics.
- `pilot/evaluation.json`: held-out weak-window metrics and dense-expansion gate.

No extra system instruction, token shortening, node batching, changed shortlist,
or quantization is silently introduced. The model stays loaded, gradients are
disabled, video decoding is shared within each video, and all later work uses cache.
There can still be up to **37 model generations per window**, so local 8B does not
automatically mean a fast full-dataset run. Measure the first completed videos.

## 4. Dense Evaluation Only if the Local Pilot Gate Passes

```bat
python -B run_local.py --stage 3 --action plan --tag v919_local_qwen8b_seed20260917 --test-root G:\Dataset\XDViolence\videos\videos --annotations G:\Dataset\XDViolence\videos\annotations_uniform_format.txt
```

This keeps the existing gate. Read `dense/plan.json`, then supply **new reviewed local
compute caps** for Stage 3; do not reuse the 400-window caps for all 800 test videos:

```bat
python -B run_local.py --stage 3 --action run --tag v919_local_qwen8b_seed20260917 --test-root G:\Dataset\XDViolence\videos\videos --annotations G:\Dataset\XDViolence\videos\annotations_uniform_format.txt --approve-local-compute --approved-by researcher --max-attempts YOUR_REVIEWED_CAP --max-reserved-output-tokens YOUR_REVIEWED_TOKEN_CAP
```

The two `YOUR_...` values are intentional placeholders. The current API arm is not
eligible for Stage 3 under its frozen local-validity gate; a local run is evaluated
separately and must not bypass its gate. Negative or inconclusive outcomes are results.

## Optional: Local Qwen3-8B Text Teacher

Supply an ordinary JSON message array in `messages.json`, for example:

```json
[{"role":"user","content":"Return JSON describing a normal visual counterexample to people standing close together."}]
```

```bat
python llm_local.py --model-dir G:\Qwen3-8B --messages messages.json --out llm_example.json
```

Add `--thinking` for Qwen3 thinking mode. Raw reasoning and final text are both retained.
This utility does not insert graphs, change scores or authorize discovery. It is a
local replacement interface for a text teacher, **not a new LLM stage in V919**.

## Interpretation Limits

This is a **backend/model comparison**, not a clean parameter-count ablation:
Qwen3.6-plus and Qwen3-VL-8B-Instruct differ in training/version, provider preprocessing,
reasoning behavior and possibly hidden inference settings. The API arm used substantial
reasoning tokens; this Instruct model does not reproduce that reasoning policy.
Keep graph catalog, cohort/roles, labels, prompts, frame content and nominal pixel/token
budgets fixed. Compare validity/coverage, fallback, AP and paired uncertainty, not only
successful examples. Report local runtime and GPU peaks separately from API latency.

The pilot has weak positive anchors and filename-A negatives; context rows are not
normal labels. Its AP is not full-test frame AP. No model-size improvement is promised.
Only a complete dense evaluation can produce the stated full-dataset frame metrics.

## Verification

```bat
python -m pytest tests -q
```

Offline tests cover model-isolated cache identities, no remote fallback, role/manifest
preservation, scoped API recovery, and missing-evidence behavior. Actual GPU inference
and numerical agreement across devices must be checked on the destination hardware.
