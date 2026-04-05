# TensorRT LoRA Refit

Swap LoRA weights (any combination, any strength) into a pre-built TRT engine in ~13 seconds instead of rebuilding from scratch (~5-10 minutes).

## How It Works

The `TensorRT Refit Loader` node updates engine weights in-place from a source model (e.g. checkpoint + LoRA). The result is visually identical to building an engine from scratch with the LoRA applied.

### Weight Mapping

TRT's internal weight names don't match PyTorch state_dict keys. Three categories exist:

| Category | Example | Count (SDXL) | Mapping |
|----------|---------|--------------|---------|
| Named UNet weights | `unet.input_blocks.0.0.weight` | ~866 | Strip `unet.` prefix |
| Fused constants | `/unet/input_blocks.0/...` | ~445 | Scalar constants from TRT optimization — no source needed |
| ONNX internal | `onnx::MatMul_15396` | ~814 (722 MatMul) | ONNX weight map sidecar |

The 722 MatMul weights are the attention projections (to_q, to_k, to_v, to_out, ff.net) — exactly the weights LoRAs target. During engine build, a `.weight_map.json` sidecar is saved that maps each `onnx::MatMul_*` name to its PyTorch key by parsing ONNX node names (e.g. `/unet/input_blocks.4.1/transformer_blocks.0/attn1/to_q/MatMul` → `input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight`).

**MatMul transposition:** PyTorch `nn.Linear` stores weights as `[out_features, in_features]`, but ONNX MatMul expects `[in_features, out_features]`. All 2D MatMul weights are transposed during refit.

## Measured Performance (SDXL Pony, RTX 4060 Ti 16GB)

| Operation | Time |
|-----------|------|
| Engine build (one-time, static 1024x1024) | ~5-10 min |
| LoRA refit (788 weights) | ~13 sec |
| Engine deserialization | ~4 sec |
| Weight extraction + mapping | ~8 sec |
| TRT refit API call | ~1 sec |

## Persistence

### How long do refitted engines persist?

Refitted engines are cached to disk in a `.refit_cache/` directory alongside the base engine. Each LoRA configuration gets its own cached engine file (keyed by a deterministic hash of the patches). After the first refit, subsequent runs with the same LoRA config skip refitting entirely — even after VRAM eviction or restart.

### What happens on VRAM eviction?

When ComfyUI's memory manager evicts the TRT engine (e.g. to load a VAE), the refitted engine is reloaded from the `.refit_cache/` file on the next run. No re-refitting needed — just a ~2-3s deserialize.

### What about disk space?

The `.refit_cache/` directory contains a copy of the refitted engine (~5 GB for SDXL). Disk management evicts refit cache files first (expendable, ~13s to rebuild) before touching base engines (5-10 min to rebuild).

## Setup (one-time per engine)

1. Load your base checkpoint (**no LoRA**)
2. Connect to **Dynamic/Static TRT Conversion** with `enable_refit: True`
3. Build — takes 5-10 minutes, but only happens once

The resulting `.engine` and `.weight_map.json` files are permanent and survive reboots.

**Context length:** Set `context_len` to cover your longest prompt (e.g. 4 = 308 tokens for SDXL with prompt weighting). Context is always dynamic — shorter prompts work fine, but exceeding the limit will error. Changing context_len requires a rebuild.

## Usage (per-LoRA, fast)

### Option A: TensorRT Loader Auto (recommended)

1. Load checkpoint → Load LoRA(s) → **MODEL** output
2. Connect MODEL to **TensorRT Loader Auto** with `refit=True`
3. First run: auto-builds engine (5-10 min) then refits (~13s)
4. Subsequent runs: loads cached engine, skips refit if LoRAs unchanged
5. Change LoRA/strength/stack → rerun → ~13 seconds for refit

The Auto node handles building, loading, matching, and caching automatically.

### Option B: TensorRT Refit Loader (manual)

1. Load checkpoint → Load LoRA(s) → **MODEL** output
2. Connect MODEL to **TensorRT Refit Loader** (`source_model`)
3. Select your engine file and model type
4. Run — refit takes ~13 seconds, then generates at TRT speed
5. Change LoRA/strength/stack → rerun → ~13 seconds again

CLIP and text encoding still use the normal LoRA pipeline — only the UNet is TRT-accelerated.

## XY Plots / Batch Sampling

TRT engines stay hot during XY plot iterations. ComfyUI's memory manager evicts models via `free_memory()` during temporary swaps (e.g. loading VAE evicts UNet, then UNet reloads). The TRT lifecycle hooks detect this isn't a full unload and skip engine teardown — engines remain in VRAM and reload instantly with no deserialization cost.

Full unloads ("Clear All Models" button, LoRA cache miss in Auto loader) go through `unload_all_models()`, which sets a force flag that ON_DETACH checks. Only then are engines actually freed from VRAM.

### VRAM lifecycle

| Scenario | ON_DETACH behavior | Cost |
|----------|-------------------|------|
| XY plot model swap | Keep hot | 0s (instant) |
| Re-queue same prompt | Keep hot | 0s |
| Clear All Models | Unload | 0s (immediate VRAM release) |
| LoRA change (Auto) | Unload (cache miss path) | ~2-4s if cached, ~13s if new |

### Refit cache (multi-slot, hash-based)

The refit cache uses a **deterministic hash** of the LoRA patches (key names, strengths, tensor fingerprints) instead of ComfyUI's random `patches_uuid`. This enables multi-slot caching:

- **Same LoRA, re-queue**: cache hit, no refit (instant)
- **LoRA A → B → A**: cache hit on return to A (~2-4s deserialize instead of ~13s refit)
- **Disk cache**: multiple files per base engine (`<engine>_<hash>.engine`), one per LoRA config
- **Memory + disk**: in-memory cache for VRAM-hot engines, disk cache survives restarts

Disk cache files are managed by the existing FIFO eviction system — refit cache files are purged first (expendable, ~13s to rebuild) before base engines.

## Limitations

- **Static dimensions** — engine is locked to the resolution and batch size it was built with (or the dynamic range, for dynamic engines). Context length is always dynamic.
- **Refit flag required** — engine must be built with `enable_refit: True`. Slightly larger and marginally slower than non-refit engines due to fewer weight fusion optimizations
- **Weight-space LoRAs only** — standard LoRA that patches linear/conv layers. Anything that changes graph topology (new layers, different architecture) won't work
- **Only SDXL tested** — the mapping logic is generic, but SD1.5/SD3/Flux/etc. are untested
- **Only static engines tested** — dynamic profile engines are untested

## FAQ

### Is a refit-enabled engine slower than a normal one?

The `REFIT` builder flag prevents certain weight fusion/folding optimizations since weights must remain individually addressable. In practice the difference is single-digit percent.

### Can I use a refit-enabled engine without refitting?

Yes. The regular `TensorRT Loader` runs it fine — it doesn't care about the refit flag. One engine works both ways: directly (base weights) or via `TensorRT Refit Loader` (with LoRA).

### Does adding/removing LoRAs work?

Yes. Stack multiple LoRAs, adjust strengths, remove some — the engine architecture is fixed but all weights are replaceable. With no LoRA applied, the refit loader writes back base weights.

### What are the 445 unmapped TRT weights?

These are scalar constants created by TRT's optimization passes (fused bias terms, constants folded into kernels). They don't correspond to meaningful model weights and don't need updating.
