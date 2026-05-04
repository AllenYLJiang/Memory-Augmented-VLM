# structural-vlm-binary

A minimal video anomaly detector that uses **direct VLM prompting with structural graphs**.

Core idea:
- Do **not** run a large multi-stage subtype pipeline.
- Prompt the VLM directly on a segment.
- Ask it to match the segment against a small library of **shared structural graphs**.
- Graphs contain both:
  - **spatial composition**: multiple cues/nodes jointly forming one event
  - **temporal/causal chain**: earlier cues leading to later cues
- Final output is **binary abnormal vs normal**.
- Optional `96x3` / `96x5` rescue updates the **center window graph posterior**, not the label.

## Install

```bash
pip install -e .
```

Set your DashScope API key:

```bash
export DASHSCOPE_API_KEY=...
```

## Predict one segment

```bash
python -m structural_vlm_binary predict-segment \
  --video-path /path/to/segment.mp4
```

## Predict a window from a full video

```bash
python -m structural_vlm_binary predict-window \
  --video-path /path/to/video.mp4 \
  --start-frame 0 --end-frame 95
```

## Evaluate videos with sliding windows

```bash
python -m structural_vlm_binary eval-videos \
  --videos-root /path/to/videos \
  --annotations /path/to/annotations.txt \
  --out-dir ./runs \
  --window 96 --stride 32 --fresh-run
```

## Output

Each prediction record includes:
- `binary_decision`
- `best_abnormal_graph`
- `best_normal_graph`
- `graph_matches`
- `observations`
- `rescue`
- runtime metadata

This project intentionally keeps the online path simple:

**segment -> direct structural graph prompt -> binary decision**

with optional wider-context rescue only for uncertain cases.


## Parallel evaluation

`eval-videos` and `eval-segments` now support:

- `--async-workers N` for concurrent inference
- `--max-consecutive-errors K` to retry retryable connection/service failures for a single item up to K times before aborting
- `--shuffle-seed S` to deterministically shuffle evaluation order before applying `--limit`

Example:

```bash
structural-vlm-binary eval-videos \
  --videos-root /path/to/videos \
  --annotations /path/to/annotations.txt \
  --out-dir ./runs \
  --async-workers 6 \
  --max-consecutive-errors 5 \
  --shuffle-seed 123
```
