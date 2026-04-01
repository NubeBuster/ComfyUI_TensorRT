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

The refitted engine lives in VRAM until evicted by ComfyUI's memory manager (e.g. when loading a different model). The `.engine` file on disk is unchanged — it keeps the original base weights. Each workflow execution re-refits (~13 seconds).

### Can the models be evicted to RAM instead of the void?

Not yet — VRAM eviction currently destroys the refitted engine. Keeping it in RAM to survive eviction is planned.

### Can refitted models be saved to and reloaded from disk?

Not yet. Saving a refitted engine as a new `.engine` file (loadable without re-refitting) is planned.

## Setup (one-time per engine)

1. Load your base checkpoint (**no LoRA**)
2. Connect to **Dynamic/Static TRT Conversion** with `enable_refit: True`
3. Build — takes 5-10 minutes, but only happens once

The resulting `.engine` and `.weight_map.json` files are permanent and survive reboots.

**Important:** Build with a context length that covers your longest prompt. SDXL with prompt weighting (e.g. A1111-style `(word:1.2)`) can produce up to 308 tokens (77x4). If the engine was built with context_len=77 and you send 308 tokens, it will fail with a shape mismatch error.

## Usage (per-LoRA, fast)

1. Load checkpoint → Load LoRA(s) → **MODEL** output
2. Connect MODEL to **TensorRT Refit Loader** (`source_model`)
3. Select your engine file and model type
4. Run — refit takes ~13 seconds, then generates at TRT speed
5. Change LoRA/strength/stack → rerun → ~13 seconds again

CLIP and text encoding still use the normal LoRA pipeline — only the UNet is TRT-accelerated.

## Limitations

- **Static dimensions** — engine is locked to the resolution, batch size, and context length it was built with (or the dynamic range, for dynamic engines)
- **Refit flag required** — engine must be built with `enable_refit: True`. Slightly larger and marginally slower than non-refit engines due to fewer weight fusion optimizations
- **Weight-space LoRAs only** — standard LoRA that patches linear/conv layers. Anything that changes graph topology (new layers, different architecture) won't work
- **Only SDXL tested** — the mapping logic is generic, but SD1.5/SD3/Flux/etc. are untested
- **Only static engines tested** — dynamic profile engines are untested
- **Re-refits on every run** — the refitted engine is not persisted to disk or RAM yet

## FAQ

### Is a refit-enabled engine slower than a normal one?

The `REFIT` builder flag prevents certain weight fusion/folding optimizations since weights must remain individually addressable. In practice the difference is single-digit percent.

### Can I use a refit-enabled engine without refitting?

Yes. The regular `TensorRT Loader` runs it fine — it doesn't care about the refit flag. One engine works both ways: directly (base weights) or via `TensorRT Refit Loader` (with LoRA).

### Does adding/removing LoRAs work?

Yes. Stack multiple LoRAs, adjust strengths, remove some — the engine architecture is fixed but all weights are replaceable. With no LoRA applied, the refit loader writes back base weights.

### What are the 445 unmapped TRT weights?

These are scalar constants created by TRT's optimization passes (fused bias terms, constants folded into kernels). They don't correspond to meaningful model weights and don't need updating.
