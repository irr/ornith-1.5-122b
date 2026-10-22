# ornith-1.5-122b

One command to run [Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B)
with llama.cpp (Vulkan/RADV) on an AMD Strix Halo (gfx1151) machine with 128 GB
unified memory, at the model's full native **262144-token** context, sped up by
MTP self-speculation (or, optionally, the [DFlash2 drafter](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2)).

`./serve-ornith.sh` downloads the GGUF weights (if missing) into `./models/` and
starts `llama-server` in Docker with an OpenAI-compatible API on
**http://localhost:8000**.

## Requirements

- AMD Strix Halo APU (gfx1151) with 128 GB unified memory, GTT sized to use it
  (`amdgpu.gttsize=131072 ttm.pages_limit=33554432` on the kernel command line)
- ~81 GB free disk for the default `UD-Q4_K_XL` quant (~95 GB for `UD-Q5_K_XL`; +~1 GB for the DFlash2 drafter)
- Docker Engine, with `/dev/kfd` and `/dev/dri` accessible
- [uv](https://docs.astral.sh/uv/), used only to run the Hugging Face CLI for the download

## Quick start

```bash
./serve-ornith.sh
curl -s http://localhost:8000/v1/models
```

## Why llama.cpp

The BF16 checkpoint is ~249 GB, which doesn't fit in 128 GB, and there's no
FP8/AWQ build that vLLM or SGLang could load on gfx1151. Unsloth's GGUF quants
fit, and the `kyuz0/amd-strix-halo-toolboxes` image ships a llama.cpp build that
supports this architecture (Qwen3.5 MoE: hybrid Gated DeltaNet + gated attention).

## What `serve-ornith.sh` does

1. **Downloads / verifies the weights** with
   `uv run --with "huggingface_hub[cli]" hf download unsloth/Ornith-1.5-122B-GGUF --include "<QUANT>/*" --local-dir models`.
   With `SPEC=dflash` it also downloads the drafter, `Ornith-1.5-122B-DFlash2-<DRAFT_QUANT>.gguf`,
   from [`kaivoss/Ornith-1.5-122B-DFlash2-GGUF`](https://huggingface.co/kaivoss/Ornith-1.5-122B-DFlash2-GGUF)
   into `models/dflash2/`.
   Interrupted downloads resume; nothing is re-fetched once complete.
2. **Resolves GPU group GIDs numerically** from `/dev/kfd` and `/dev/dri/*`.
   Docker Engine 29 can't resolve host group names like `render` inside the image.
3. **Starts `llama-server`** (`docker run --rm`), with `./models` mounted read-only
   at `/models`, every layer on the GPU, flash attention, 4 parallel slots sharing one 262144-token KV cache,
   context checkpoints plus a RAM prompt cache (so long prompts can be reused),
   reasoning returned in `reasoning_content`, and the model card's sampling defaults.
   Speculative decoding uses the model's built-in MTP head by default. `SPEC=dflash` uses the
   DFlash2 drafter on the GPU instead (`--spec-type draft-dflash`, 4 draft tokens per step).
   Either way, output is identical to the target model alone; only decoding speed changes.

## MTP vs DFlash2 (measured on Strix Halo, UD-Q4_K_XL)

| Setup                     | Generation | Draft tokens accepted |
| ------------------------- | ---------- | --------------------- |
| `SPEC=mtp` (default)      | ~30 tok/s  | ~92%                  |
| `SPEC=dflash`, 4 drafts   | ~26 tok/s  | ~39%                  |
| `SPEC=dflash`, 15 drafts  | ~9 tok/s   | ~9%                   |

Short (~800-token) prompt, llama.cpp build 11783 (DFlash 15-draft run on build 11206). The DFlash2
drafter accepts far fewer tokens here than its model card reports, so MTP stays the default.

## Configuration

Set any of these as environment variables:

| Variable         | Default                                       | Purpose                                              |
| ---------------- | --------------------------------------------- | ---------------------------------------------------- |
| `QUANT`          | `UD-Q4_K_XL`                                  | GGUF quant to download and serve                     |
| `MODE`           | `coding`                                      | `general` or `coding` sampling preset (model card)   |
| `SPEC`           | `mtp`                                         | `mtp` (built-in MTP), `dflash` (DFlash2 drafter), or `none` |
| `DRAFT_REPO`     | `kaivoss/Ornith-1.5-122B-DFlash2-GGUF`        | GGUF repo of the DFlash2 drafter                     |
| `DRAFT_QUANT`    | `Q8_0`                                        | Drafter quant (`Q8_0` or `Q6_K` recommended)         |
| `DRAFT_N_MAX`    | `4`                                           | DFlash draft tokens per step (max 15)                |
| `VISION`         | `1`                                           | Loads `mmproj-F16.gguf` for image input; `0` disables |
| `KV_TYPE`        | `f16`                                         | KV cache type; `q8_0` halves it                      |
| `CTX`            | `262144`                                      | Total context, pooled across all slots (`--kv-unified`) |
| `PARALLEL`       | `4`                                           | Number of slots (concurrent requests)                |
| `PORT`           | `8000`                                        | Host port the API is published on                    |
| `IMAGE`          | `kyuz0/amd-strix-halo-toolboxes:vulkan-radv`  | Server image                                         |
| `CONTAINER_NAME` | `ornith-1.5-122b`                             | Name for `docker run --name`                         |
| `MODELS_DIR`     | `./models`                                    | Where the weights live (gitignored)                  |

## Memory

| Item                                  | Size       |
| ------------------------------------- | ---------- |
| Weights, `UD-Q4_K_XL`                 | ~80 GB     |
| KV cache, f16, 262144 tokens          | ~6 GiB     |
| DFlash2 drafter, Q8_0 (`SPEC=dflash`) | ~1–2 GB    |
| Linear-attention state + compute bufs | a few GB   |
| **Total** (default, `SPEC=mtp`)       | **~90 GB** |

Only 12 of the 48 layers use full attention (2 KV heads, head dim 256), so the
KV cache stays small even at the full context. `UD-Q5_K_XL` needs about 104 GB,
which fits but leaves little for the OS.

## Sampling (model card)

- General: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5`
- Precise coding: `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0`

Clients can override these per request.
