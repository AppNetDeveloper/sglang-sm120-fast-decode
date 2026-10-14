# Faster low-concurrency decode for Qwen3.8-27B on RTX PRO 6000 (SM120) with SGLang

Three small engine-side changes for **SGLang 0.5.21** on a single **RTX PRO 6000 Blackwell (SM120)**:
**+22 % to +38 % output tokens/s in production traffic** (1–15 concurrent requests), **+34 % single-stream**
at 45k context in a controlled bench, **same quality**, **no extra memory** (KV pool and max running
requests unchanged). Measured on 14 Oct 2026 with the `ThinkingCap-Qwen3.8-27B-NVFP4A4-AWQ` checkpoint
(compressed-tensors: NVFP4 W4A4 MLPs, FP8 per-channel attention/GDN projections and lm_head, bf16 MTP head),
MTP/EAGLE 3/1/4, FP8 KV cache, HiCache.

## Why
A torch-profiler trace of single-stream decode (≈30.7 ms per scheduler step, GPU 99 % busy) showed the
step is dominated by weight-streaming GEMMs that run far below the card's ~1.79 TB/s at M = 1–16 tokens:

| Part of the step | Time | Effective bandwidth |
|---|---|---|
| FP8 per-token GEMMs (CUTLASS) | 10.6 ms | ~46–54 % |
| NVFP4 W4A4 GEMMs (FlashInfer CUTLASS) | 7.7 ms | ~50–70 % |
| Draft lm_head (full 248,320-row vocab, 3×/step) | 2.9 ms | ~74 % |

## What is in here
| Change | Effect | Files |
|---|---|---|
| **FR-Spec with a quantized lm_head.** `--speculative-token-map` crashes (OOM) when the target lm_head is FP8: SGLang keeps the FP8 weight transposed `[K, V]`, so `head.data[hot]` slices K instead of the vocab and the per-row scale is left full size. The patch builds a pruned copy that slices weight **and** scale along the vocab axis, same dtype/layout; it refuses (clear error) layouts it cannot slice per row. Draft lm_head: ~1.0 ms → ~0.18 ms per call, bit-identical logits on the kept rows. | `patches/frspec_fp8_lmhead.patch` |
| **KDA NVFP4 SM120 GEMM for more shapes.** SGLang already ships an SM120 NVFP4 kernel that reaches 85–89 % of bandwidth at small M (×1.27–1.71 vs the default), but only for an allowlist of shapes. The patch lets you add shapes via `SGLANG_KDEV_KDA_NVFP4_KN="KxN,..."` and `SGLANG_KDEV_KDA_NVFP4_MAX_M`. | `patches/kda_nvfp4_allowlist.patch` |
| **Tuned `triton_scaled_mm` configs for FP8 per-token GEMMs on SM120.** SGLang reads `N=..,K=..,device_name=..,dtype=fp8_w8a8_channelwise.json`; a 16×32×256 / 4 warps / 4 stages tile reaches ~87 % (×1.88 vs CUTLASS) at M ≤ 16. Entries are `null` from M = 24 up so prefill keeps CUTLASS. | `fp8_configs/*.json` → `<sglang>/kernels/ops/gemm/configs/` |
| **Hot-token list for the draft** (40,000 ids, multiple of 64; built only from public lists — HyperQwen + MiaAI-Lab es_en_code_65k — ranked by frequency on real Spanish/English agent traffic). The Qwen3.8 27B and Flash-Next tokenizers are byte-identical. | `tokens/` |
| Tools: GEMM microbench in CUDA graphs with real shapes (writes the JSONs), FR-Spec equivalence test, trace summarizer. | `tools/` |

## Results (same server, same bench, 400 output tokens, unique prefixes)
| | baseline | FR-Spec only | all three |
|---|---|---|---|
| 1 req, 45k ctx | 98.5 tok/s | 102.0 | **131.7 (+34 %)** |
| 4 req, 45k ctx | 45.9 | 49.7 | **52.1 (+13.5 %)** |
| 8 req, 27k ctx | 38.7 | 38.9 | 40.0 (+3 %) |
| 1 req, 141k ctx | 93.9 | 104.7 | **116.6 (+24 %)** |

Quality: reasoning 48/48 (3 runs × 16) for both baseline and patched, tool calls 24/24, needle 4/4; MTP accept
length ~3.0 in both. One hour of real production traffic (gateway logs, responses ≥ 300 tokens, bucketed by
context size): +15 % to +38 % tokens/s per call in every bucket, 0 server errors.

At 8+ concurrent requests the gain shrinks because attention over long contexts becomes the larger cost.

## How to apply
```bash
SG=$(python -c "import sglang,os;print(os.path.dirname(sglang.__file__))")   # copy it first if you prefer
cd "$(dirname "$SG")" && patch -p0 < /path/to/patches/frspec_fp8_lmhead.patch && patch -p0 < /path/to/patches/kda_nvfp4_allowlist.patch
cp /path/to/fp8_configs/*.json "$SG/kernels/ops/gemm/configs/"   # only valid for NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition and these N,K
export SGLANG_KDEV_KDA_NVFP4_KN="5120x34816,17408x5120" SGLANG_KDEV_KDA_NVFP4_MAX_M=16 SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE=1
python -m sglang.launch_server ... --speculative-token-map /path/to/tokens/hot_tokens_qwen38_es_en_code_40000.pt
```
Check the server log for `FR-Spec: draft lm_head pruned to 40000 rows`, `Using configuration from .../N=...json`
and `KDA NVFP4 SM120 path`. For another GPU or checkpoint, regenerate configs with
`tools/microbench.py --solo fp8 --m 1,4` (set `FP8_CFG_INSTALL` to auto-install winners) and validate KDA with `--solo fp4`.

## Caveats
- Tested only on SGLang 0.5.21, RTX PRO 6000 Workstation Edition (600 W), CUDA 13, this checkpoint. The JSONs are per (N, K, GPU name).
- The token list must be multiple of 16 (64 recommended) or the pruned FP8 lm_head falls back to an untuned kernel.
- Things that did **not** help here: `--linear-attn-prefill-backend flashinfer`, `--enable-torch-compile`, `--bf16-gemm-backend gemv` (requires SM90), `--fp8-gemm-backend` (only affects block-wise FP8).

Apache-2.0. Contributions welcome via issues/PRs; the maintainers review every change.
