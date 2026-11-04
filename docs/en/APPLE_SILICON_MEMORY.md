# Apple Silicon: where the memory goes, and how to use less

Measured notes for running the REST API server (`start_api_server_macos.sh`, MLX backend) on a Mac with
32 GB of unified memory. Everything below was measured on one machine; numbers are footprints as reported
by macOS `footprint -p <pid>` (resident + compressed + swapped), not RSS.

| | |
|---|---|
| Machine | Apple M1 Pro, 32 GB, macOS 26.3 |
| ACE-Step | this fork, branch `feat/apple-silicon-memory` (based on upstream `main`, 2026-11) |
| What ACE-Step sees | `macOS MPS detected (25.0 GB unified memory, tier=unlimited)`: macOS lets the GPU use ~75 % of the RAM |

## Summary

- A 4.5 GB turbo checkpoint costs **20 GB** after loading; `acestep-v15-xl-turbo` (18.6 GB on disk) peaks at
  **37 GB** while loading and pushed ~18 GB into swap on this Mac.
- Four things multiply the weights: float32 on MPS (2x), a second native MLX copy of the DiT decoder (2x),
  the load peak (a float32 copy of the whole decoder is held while converting), and a language model that
  is loaded by default.
- Three opt-in switches and a cache release bring turbo to **13 GB** and xl-turbo to **23 GB** with swap flat,
  and roughly halve the load time. Defaults are unchanged.

## Where the memory goes (turbo, language model off)

| Phase | Footprint |
|---|---|
| Server idle | 0.4 GB |
| After the models load (lazily, on the first request) | 20 GB (does not grow while generating) |

Decomposition after loading (from `torch.mps.current_allocated_memory()`, `mx.get_active_memory()` and the
parameter sizes):

| Component | Size |
|---|---|
| PyTorch DiT, float32 (decoder 6.3 + the rest 3.3) | 9.6 GB |
| PyTorch text encoder (Qwen3-Embedding-0.6B), float32 | 2.4 GB |
| PyTorch VAE, float32 | 0.3 GB |
| MLX copy of the DiT decoder, float32 | 6.3 GB |
| MLX VAE, float32 | 0.7 GB |
| Total | ~19-20 GB |

Without the MLX DiT (`use_mlx_dit=False`) the footprint is 14 GB, so the MLX copy costs ~6 GB.

## Why

1. **float32 on MPS.** `init_service_orchestrator.py` loaded everything in float32 for any device that is not
   CUDA, ROCm or XPU (`self.dtype = torch.bfloat16 if resolved_device == "xpu" else torch.float32`). The
   checkpoints are bf16 on disk, so every PyTorch weight takes twice the space. (The line predates the March
   2026 commit `de512c0` that restructured it; no reason is documented.)
2. **Second copy, also float32.** `acestep/models/mlx/dit_convert.py` (`np_val = value.detach().cpu().float().numpy()`)
   converts the decoder to an MLX array per tensor but keeps **all** of them as float32 in a list before
   loading. During conversion the PyTorch copy, the float32 list and (with the bf16 flag of upstream PR #1251)
   the bf16 cast are alive at once. That is the load peak: 19 GB for turbo, 37 GB for xl-turbo.
3. **Allocator caches.** After the bf16 cast ~6.3 GB stays in MLX's buffer cache and counts against the
   process until `mx.clear_cache()` / `torch.mps.empty_cache()` is called.
4. **A language model you may not need.** For GPUs with >= 24 GB (`gpu_config.py`, tier `"unlimited"`)
   `init_lm_default` is true and the recommended LM is `acestep-5Hz-lm-4B`; `startup_llm_init.py` picks it
   unless `ACESTEP_LM_MODEL_PATH` is set. With `thinking=false` the LM is not used for audio codes, but
   `use_cot_caption`, `use_cot_language` and `use_cot_metas` default to true, so it still runs on **every**
   request (`[generate_music] LLM usage decision: thinking=False, use_cot_caption=True, ...` followed by
   `Phase 1: Generating CoT metadata...`). Switching it off (`ACESTEP_INIT_LLM=false`) saved 7-9 GB and about
   75 s per generation (turbo, 10 s song: 105-175 s with the LM, 30 s without, cold). Without the LM
   your caption is used as written instead of being rewritten, which changes the output; judge it by ear.

## What changed on this branch (all opt-in)

| Switch | Effect |
|---|---|
| `ACESTEP_MPS_BF16=1` | PyTorch parts load in bfloat16 on MPS (`apple_precision.non_cuda_dtype`) |
| `ACESTEP_MLX_DIT_BF16=1` | native MLX DiT in bfloat16 (from upstream PR #1251) **and** conversion straight to bf16, one tensor at a time (`convert_decoder_weights(..., dtype=)`); no float32 copy of the decoder is held |
| (always, on MPS) | caches are emptied after loading (`apple_precision.release_unused_device_memory`) |
| `ACESTEP_MLX_VAE_CACHE_MB` | cap on MLX's buffer cache while the VAE decodes (from upstream PR #1362, default 4096, `0` = no cap); bit-identical audio |
| `ACESTEP_FREE_PYTORCH_DECODER=1` | the PyTorch DiT decoder stays on the CPU, the MLX decoder is converted from it and the copy is dropped, so it never reaches the GPU (-3 GB turbo, -7 GB xl-turbo). `get_lyric_timestamp` / `generate_lrc` then answers with an error; if MLX does not start, the decoder is moved to the device after all |
| `ACESTEP_INIT_LLM=false` | `start_api_server_macos.sh` no longer overwrites a value given from outside |

Results, same seed, 20 s song with lyrics:

| | Load peak | Steady | Load time |
|---|---|---|---|
| turbo, defaults | 19 GB | 14 GB* | 13 s |
| turbo, three switches | 13 GB | 13 GB | 7 s |
| xl-turbo, bf16 patches applied after the fact | 37 GB (swap +17.8 GB) | 23 GB | ~90 s |
| xl-turbo, three switches | 23 GB (swap flat) | 23 GB | 33 s |

\* with the caches emptied by hand; without it 20 GB. xl-turbo: a 20 s song with lyrics in 70 s, diffusion
0.88 s/step (turbo ~0.27 s/step).

The decoded audio of the xl-turbo run is bit-identical (same PCM checksum) to a run where the same patches
were applied from outside, and the lyric alignment (`generate_lrc`) is identical. bf16 against float32 is
close but not identical: correlation 0.978 (10 s, no lyrics) and 0.997 (20 s, with lyrics), no NaN, level
within a few dB in both directions. Whether it sounds as good is a listening question.

## The memory-pressure peak is the VAE decode

A steady footprint hides it. Sampling every second while generating (xl-turbo, 20 s song) showed 23 GB steady, then **35 GB** exactly during
`[MLX-VAE] Decoded 1 sample(s), 500 latent frames` (float32), with the macOS pressure level at 4 (critical) and swap growing 6 -> 15 GB. The decode
holds 12 GB (20 s song) to 20 GB (60 s) in MLX's buffer cache on top of the model.

| Same seed, lean settings | Peak | Pressure | Swap growth | Total | Audio |
|---|---|---|---|---|---|
| turbo 60 s, no cap | 34 GB | 4 | +8.4 GB | 46 s | reference |
| turbo 60 s, `ACESTEP_MLX_VAE_CACHE_MB=1024` | 20 GB | 1 | 0 | 31 s | identical |
| turbo 60 s, + `ACESTEP_MLX_VAE_FP16=1` | 17 GB | 1 | 0 | 26 s | correlation 0.9991 |
| xl-turbo 20 s, no cap | 35 GB | 4 | +8.3 GB | 72 s | reference |
| xl-turbo 20 s, cap 1 GiB | 28 GB | 4 | +4.6 GB | 61 s | identical |
| xl-turbo 20 s, cap 1 GiB + fp16 VAE | 25 GB | 2 | +0.3 GB | 51 s | correlation 0.9989 |
| xl-turbo 60 s, cap 1 GiB + fp16 VAE | 26 GB | 2 | +2.6 GB | 71 s | |

The cap is the main fix and also makes the decode faster. A smaller decode window (`ACESTEP_MLX_VAE_CHUNK` 192/256) did not lower the peak
(256 was worse). fp16 in the VAE lowers the peak further and is needed to keep xl-turbo out of critical pressure; it changes the audio slightly.

## Not loading the PyTorch decoder (`ACESTEP_FREE_PYTORCH_DECODER=1`)

The MLX decoder does the diffusion; the PyTorch decoder is only used by the lyric aligner. Keeping it on the CPU for the conversion and then dropping it:

| Lean settings + fp16 VAE | Steady | Peak while loading | 
|---|---|---|
| turbo | 13 GB -> **10 GB** | 14 -> 10 GB |
| xl-turbo | 22 GB -> **15 GB** | 23 GB (checkpoint conversion) |

xl-turbo 60 s song through `start_api_server_macos.sh`: 61 s, footprint 15-19 GB after loading (also during the VAE decode), memory pressure 1 all the time, swap flat.
The price: `generate_lrc` is unavailable; callers that time lyrics another way (Whisper on the vocal stem) do not notice.

## A bf16 copy of the checkpoint, and switching models

The xl-turbo checkpoint on the Hub is float32 (829 tensors, 19 GB) even though its config says bfloat16, and ACE-Step casts to bf16 on load. Writing the bf16 copy once
(`safetensors`: cast each float32 tensor, save the four shards; 33 s, 9.3 GB) lowers the load: peak 23 -> 19 GB, file cache while loading 16.8 -> 10.7 GB, and the generated audio
is bit-identical (same PCM checksum). Point `ACESTEP_CONFIG_PATH` at the new folder.

With the lean settings, `/v1/init` model switches give the memory back: xl-turbo 15 GB -> turbo 10 GB -> xl-turbo 15 GB over four switches without growth (13-18 s each, 18-19 GB at the peak of a switch).

## Things that did not work or have limits

- **Dropping the PyTorch decoder after loading** saves only 2-3 GB (the MPS allocator keeps the freed blocks) and `get_lyric_timestamp` still calls it
  (`TypeError: 'NoneType' object is not callable`). Hence `ACESTEP_FREE_PYTORCH_DECODER`, which never puts it on the GPU, with a clear error from the aligner.
- A small MLX cache limit (`mx.set_cache_limit`) added nothing once the conversion is per tensor.
- A switch of models (`/v1/init`) in a **float32** server did not give the memory back: turbo -> sft -> xl-turbo
  reached 50 GB (39 GB in swap). In the bf16 server it does (see above).
- `ACESTEP_INIT_LLM` is also set by `start_api_server_macos.sh` itself in stock ACE-Step; an env var from
  outside was silently overwritten there (fixed on this branch).

## Measuring (do not trust `top` or RSS)

`top` showed 50 GB and `ps -o rss` 1-3 GB for the same process. Use `footprint -p <pid>`; the "graphics"
rows are Metal/MLX buffers. Inside Python: `torch.mps.current_allocated_memory()`,
`torch.mps.driver_allocated_memory()`, `mx.get_active_memory()`, `mx.get_cache_memory()`. Measure per phase
(idle, after load, after generation 1 and 2) and put a guard on swap when testing a large model.

## Related upstream issues and pull requests (ace-step/ACE-Step-1.5)

| # | Status | About |
|---|---|---|
| #1081 | closed (stale) | MLX + xl-turbo + 5Hz-4B drains memory on a 32 GB Mac mini; maintainer analysis; fix is PR #1097 (Autoscore) |
| #1097 | open | stop the Autoscore unified-memory leak on MLX |
| #1251 | open | opt-in bf16 for the native MLX DiT (included here) |
| #1362 | open | cap the MLX buffer cache during native VAE decode (swap on long tracks); confirmed on this Mac and included here |
| #1090 | open | unload the LM before the DiT phase |
| #198 | closed (stale) | the API silently loads the LM on the first request and keeps ~4 GB |
| #1336 | closed (stale) | MPS "disable upper limit for memory allocations" on an M1 Pro |
| #1350 | closed (stale) | non-turbo DiT exceeds the 600 s `ACESTEP_GENERATION_TIMEOUT` on one user's GPU |
| #1259, #1288 | closed | sft/base garbled audio caused by DCW defaults; fixed by #1273 and #1282 |

The float32-on-MPS default and the duplicate float32 decoder were not found in any issue.

## Reproduce

```bash
cd vendor/ACE-Step-1.5
# defaults
./start_api_server_macos.sh
# lean
ACESTEP_INIT_LLM=false ACESTEP_MPS_BF16=1 ACESTEP_MLX_DIT_BF16=1 ACESTEP_MLX_VAE_CACHE_MB=1024 ./start_api_server_macos.sh
# leanest (no lyric alignment from the server)
ACESTEP_INIT_LLM=false ACESTEP_MPS_BF16=1 ACESTEP_MLX_DIT_BF16=1 ACESTEP_MLX_VAE_CACHE_MB=1024 ACESTEP_MLX_VAE_FP16=1 ACESTEP_FREE_PYTORCH_DECODER=1 ./start_api_server_macos.sh
# another model (xl-turbo also wants the fp16 VAE on a 32 GB Mac)
ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo ACESTEP_INIT_LLM=false ACESTEP_MPS_BF16=1 ACESTEP_MLX_DIT_BF16=1 ACESTEP_MLX_VAE_CACHE_MB=1024 ACESTEP_MLX_VAE_FP16=1 ./start_api_server_macos.sh
```

Unit tests: `uv run python -m unittest acestep.core.generation.handler.apple_precision_test acestep.models.mlx.dit_convert_test acestep.core.generation.handler.mlx_dit_init_test acestep.core.generation.handler.mlx_vae_native_test`
(one test in `init_service_test` (`...cuda_sync_cleanup_error`) failed before these changes as well).
