#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${TAG:-governed_v919_effectiveness_20260917_r1}"
WORK="${WORK:-$PROJECT/runs/$TAG}"
ARGS=(--stage "${STAGE:-1}" --action "${ACTION:-plan}" --out "$WORK")
ARGS+=(--seed "${SEED:-20260917}" --pilot-videos "${PILOT_VIDEOS:-200}" --pilot-windows-per-video "${PILOT_WINDOWS_PER_VIDEO:-2}")
ARGS+=(--window "${WINDOW_FRAMES:-96}" --stride "${STRIDE_FRAMES:-48}" --workers "${ASYNC_WORKERS:-3}")
ARGS+=(--top-k-abnormal "${TOP_K_ABNORMAL_GRAPHS:-4}" --top-k-normal "${TOP_K_NORMAL_GRAPHS:-6}")
ARGS+=(--model "${VLM_MODEL:-qwen3.6-plus}" --max-output-tokens "${MAX_OUTPUT_TOKENS:-8192}" --image-max-pixels "${IMAGE_MAX_PIXELS:-262144}")
ARGS+=(--regularization "${REGULARIZATION:-0.1}" --bootstrap "${BOOTSTRAP:-500}")
ARGS+=(--expected-test-videos "${EXPECTED_TEST_VIDEOS:-800}")
[[ -n "${TRAIN_ROOT:-}" ]] && ARGS+=(--train-root "$TRAIN_ROOT")
[[ -n "${ANCHOR_ROOT:-}" ]] && ARGS+=(--anchor-root "$ANCHOR_ROOT")
[[ -n "${TEST_ROOT:-}" ]] && ARGS+=(--test-root "$TEST_ROOT")
[[ -n "${ANNOTATIONS:-}" ]] && ARGS+=(--annotations "$ANNOTATIONS")
[[ -n "${GRAPH_CATALOG:-}" ]] && ARGS+=(--graph-catalog "$GRAPH_CATALOG")
[[ -n "${REUSE_RUN:-}" ]] && ARGS+=(--reuse-run "$REUSE_RUN")
[[ "${RETRY_TRANSPORT_FAILURES:-0}" == 1 ]] && ARGS+=(--retry-transport-failures)
ARGS+=(--transport-max-attempts "${TRANSPORT_MAX_ATTEMPTS:-3}" --retry-base-seconds "${RETRY_BASE_SECONDS:-10}" --retry-jitter-seconds "${RETRY_JITTER_SECONDS:-3}")
ARGS+=(--approved-by "${APPROVED_BY:-}")
if [[ "${APPROVE_BUDGET:-0}" == 1 ]]; then
  ARGS+=(--approve-budget --max-attempts "${MAX_ATTEMPTS:-0}" --max-reserved-output-tokens "${MAX_RESERVED_OUTPUT_TOKENS:-0}")
fi
export PYTHONPATH="$PROJECT/tools:$PROJECT/docs${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
echo "[v919] stage=${STAGE:-1} action=${ACTION:-plan}; output=$WORK"
if [[ "${ACTION:-plan}" == upgrade ]]; then
  echo '[v919] offline runtime compatibility audit only; no API; frozen protocol and caches preserved'
else
  echo '[v919] new authorization only; no discovery/verifier/C0; human overlays never enter scoring'
  [[ "${ACTION:-plan}" != run ]] || echo "[v919] transport_retry=${RETRY_TRANSPORT_FAILURES:-0}; cumulative attempts/request=${TRANSPORT_MAX_ATTEMPTS:-3}; delays approximately 10s/30s at defaults"
fi
exec "${PYTHON:-python}" -B "$PROJECT/tools/effectiveness_v919_cli.py" "${ARGS[@]}"
