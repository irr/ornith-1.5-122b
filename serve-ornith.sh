#!/usr/bin/env bash
# Serve ornith-ai/Ornith-1.5-122B on AMD Strix Halo (Ryzen AI Max+ 395, 128 GB unified)
# with llama.cpp (Vulkan/RADV) at the full 262144-token native context.
# Downloads weights into ./models (if missing) and starts the server with Docker.
#
# Usage:
#   ./serve-ornith.sh                    # download (if needed) + serve UD-Q4_K_XL
#   QUANT=UD-Q5_K_XL ./serve-ornith.sh   # higher quality, ~104 GB resident (tight)
#   MODE=coding ./serve-ornith.sh        # card's "precise coding" sampling defaults
#   SPEC=dflash ./serve-ornith.sh        # DFlash2 drafter instead of built-in MTP self-speculation
#   SPEC=none ./serve-ornith.sh          # disable speculative decoding
#   DRAFT_QUANT=Q6_K ./serve-ornith.sh   # smaller drafter quant (Q8_0 by default)
#   VISION=0 ./serve-ornith.sh           # skip the vision projector (image input on by default)
#
# SPEC=dflash uses the DFlash2 block-diffusion drafter
# incoai/Ornith-1.5-122B-DFlash2 (0.9B), loaded from its GGUF conversion. Output is identical
# to the target model alone. Measured here it was ~15% slower than MTP (26 vs 30 tok/s), so MTP is the default.
#
# OpenAI-compatible API: http://localhost:${PORT}/v1   (model name: Ornith-1.5-122B)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${MODELS_DIR:-${SCRIPT_DIR}/models}"
REPO="${REPO:-unsloth/Ornith-1.5-122B-GGUF}"
QUANT="${QUANT:-UD-Q4_K_XL}"
DRAFT_REPO="${DRAFT_REPO:-kaivoss/Ornith-1.5-122B-DFlash2-GGUF}"   # GGUF of incoai/Ornith-1.5-122B-DFlash2
DRAFT_QUANT="${DRAFT_QUANT:-Q8_0}"                                 # Q8_0 or Q6_K recommended for drafting
DRAFT_N_MAX="${DRAFT_N_MAX:-4}"                                    # 4 measured fastest on Strix Halo (15 was ~3x slower)
DRAFT_DIR="${MODELS_DIR}/dflash2"                                  # kept apart so the target-quant search can't pick it up
IMAGE="${IMAGE:-kyuz0/amd-strix-halo-toolboxes:vulkan-radv}"
CONTAINER_NAME="${CONTAINER_NAME:-ornith-1.5-122b}"
PORT="${PORT:-8000}"          # host port to publish
API_PORT="8080"               # container-side llama-server port
CTX="${CTX:-262144}"
PARALLEL="${PARALLEL:-4}"     # concurrent requests; with --kv-unified all slots share one CTX-token KV cache
KV_TYPE="${KV_TYPE:-f16}"     # f16 KV is only ~6 GiB at 262k (12 full-attn layers x 2 KV heads); q8_0 halves it
MODE="${MODE:-coding}"       # general | coding
SPEC="${SPEC:-mtp}"           # mtp | dflash | none
case "${SPEC}" in dflash|mtp|none) ;; *) echo "Unknown SPEC=${SPEC} (use dflash, mtp or none)" >&2; exit 1 ;; esac
VISION="${VISION:-1}"         # 1 = load mmproj for image input (+0.9 GB), 0 = text only

echo "=== Ornith-1.5-122B (llama.cpp / Vulkan) ==="
echo "Models directory: ${MODELS_DIR}"
echo "Quant: ${QUANT}  Context: ${CTX}  Slots: ${PARALLEL}  KV: ${KV_TYPE}  Mode: ${MODE}  Spec: ${SPEC}"
[[ "${SPEC}" == dflash ]] && echo "Drafter: ${DRAFT_REPO} (${DRAFT_QUANT}), draft tokens: ${DRAFT_N_MAX}"
echo

# Ensure models directory exists
mkdir -p "${MODELS_DIR}"

# Download weights with uv (resumes; skips files already present)
echo "Downloading / verifying model weights..."
includes=(--include "${QUANT}/*" --include "*${QUANT}.gguf")
[[ "${VISION}" == 1 ]] && includes+=(--include "mmproj-F16.gguf")
uv run --with "huggingface_hub[cli]" hf download "${REPO}" "${includes[@]}" --local-dir "${MODELS_DIR}"
DRAFT_NAME="Ornith-1.5-122B-DFlash2-${DRAFT_QUANT}.gguf"
if [[ "${SPEC}" == dflash ]]; then
  uv run --with "huggingface_hub[cli]" hf download "${DRAFT_REPO}" --include "${DRAFT_NAME}" --local-dir "${DRAFT_DIR}"
  [[ -f "${DRAFT_DIR}/${DRAFT_NAME}" ]] || { echo "Drafter ${DRAFT_NAME} not found in ${DRAFT_REPO}" >&2; exit 1; }
fi
echo "Download complete."
echo

# First shard of a split GGUF (or the single file); llama.cpp loads the rest itself.
MODEL_FILE="$(find "${MODELS_DIR}" \( -path "${MODELS_DIR}/.cache" -o -path "${DRAFT_DIR}" \) -prune -o -name "*${QUANT}*.gguf" -print | sort | head -n1)"
[[ -n "${MODEL_FILE}" ]] || { echo "No GGUF found for ${QUANT} in ${MODELS_DIR}" >&2; exit 1; }
MODEL_IN_CTR="/models/${MODEL_FILE#"${MODELS_DIR}"/}"
echo "Model file: ${MODEL_IN_CTR}"

# Resolve GPU group GIDs numerically from the host device nodes.
# Docker Engine 29 resolves --group-add *names* against the image's /etc/group
# (which lacks host groups like 'render') and fails with:
#   "Unable to find group render: no matching entries in group file"
# Passing numeric GIDs sidesteps the name lookup entirely.
GPU_GIDS=""
for dev in /dev/kfd /dev/dri/card* /dev/dri/renderD*; do
  if [[ -e "${dev}" ]]; then
    gid="$(stat -c '%g' "${dev}")"
    [[ " ${GPU_GIDS} " == *" ${gid} "* ]] || GPU_GIDS="${GPU_GIDS:+${GPU_GIDS} }${gid}"
  fi
done
echo "GPU device group GIDs: ${GPU_GIDS:-none found}"

GPU_GROUP_ADDS=()
for gid in ${GPU_GIDS}; do
  GPU_GROUP_ADDS+=(--group-add "${gid}")
done

# Sampling defaults from the model card
if [[ "${MODE}" == coding ]]; then
  SAMPLING=(--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0)
else
  SAMPLING=(--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 1.5 --repeat-penalty 1.0)
fi

EXTRA=()
if [[ "${VISION}" == 1 ]]; then EXTRA+=(--mmproj /models/mmproj-F16.gguf); else EXTRA+=(--no-mmproj); fi
case "${SPEC}" in
  dflash) EXTRA+=(--spec-draft-model "/models/dflash2/${DRAFT_NAME}" --spec-draft-ngl all
                  --spec-type draft-dflash --spec-draft-n-max "${DRAFT_N_MAX}") ;;
  mtp)    EXTRA+=(--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75) ;;
esac

# Start the container
echo "Starting llama-server on port ${PORT}..."
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
exec docker run --rm -p "${PORT}:${API_PORT}" \
  --name "${CONTAINER_NAME}" \
  --device /dev/kfd --device /dev/dri \
  "${GPU_GROUP_ADDS[@]}" \
  --security-opt seccomp=unconfined \
  --ipc=host --ulimit memlock=-1:-1 \
  -v "${MODELS_DIR}:/models:ro" \
  -v "${SCRIPT_DIR}/chat_template.jinja:/chat_template.jinja:ro" \
  "${IMAGE}" \
  llama-server \
    --model "${MODEL_IN_CTR}" \
    --alias Ornith-1.5-122B \
    --host 0.0.0.0 --port "${API_PORT}" \
    --ctx-size "${CTX}" \
    --parallel "${PARALLEL}" \
    --kv-unified \
    --n-gpu-layers 999 \
    --flash-attn on \
    --cache-type-k "${KV_TYPE}" --cache-type-v "${KV_TYPE}" \
    --batch-size 4096 --ubatch-size 1024 \
    --load-mode none \
    --threads 16 \
    --jinja \
    --chat-template-file /chat_template.jinja \
    --reasoning-format deepseek \
    --reasoning-budget 4096 \
    --ctx-checkpoints 16 \
    --cache-ram 16384 \
    --metrics \
    "${SAMPLING[@]}" \
    "${EXTRA[@]}"
