# TensorRT LoRA Refit

Swap LoRA weights (any combination, any strength) into a pre-built TRT engine in seconds instead of rebuilding from scratch.

<!-- TODO: Uses trt.Refitter API to update engine weights in-place.
     Extracts LoRA-patched weights from source model's state_dict,
     maps PyTorch keys to TRT weight names (strips "unet." prefix
     from ONNX export wrapper), calls set_named_weights() + refit_cuda_engine().
     Refitted engine is functionally identical to a from-scratch build. -->

## How It Works

The `TensorRT Refit Loader` node updates engine weights in-place from a source model (e.g. checkpoint + LoRA). The result is identical to building an engine from scratch with the LoRA applied — same output, just faster.

## Persistence

### How long do refitted engines persist?

The refitted engine lives in VRAM until evicted by ComfyUI's memory manager (e.g. when loading a different model). The `.engine` file on disk is unchanged — it keeps the original base weights. Each workflow execution re-refits (still seconds, not minutes).

### Can the models be evicted to RAM instead of the void?

<!-- TODO: implement RAM persistence — serialize engine via engine.serialize()
     before VRAM eviction, hold bytes in CPU memory, deserialize on reload
     instead of re-refitting from disk + source model -->

Not yet — VRAM eviction currently destroys the refitted engine. Keeping it in RAM to survive eviction is planned.

### Can refitted models be saved to and reloaded from disk?

<!-- TODO: implement disk persistence — engine.serialize() to .engine file,
     then loadable via regular TensorRT Loader without re-refitting.
     Effectively "bake LoRA into permanent engine" without minutes-long rebuild -->

Not yet. Saving a refitted engine as a new `.engine` file (loadable without re-refitting) is planned.

## Setup (one-time per engine)

1. Load your base checkpoint (**no LoRA**)
2. Connect to **Dynamic/Static TRT Conversion** with `enable_refit: True`
3. Build — takes minutes, but only happens once

The resulting `.engine` file is permanent and survives reboots.

## Usage (per-LoRA, fast)

1. Load checkpoint → Load LoRA(s) → **MODEL** output
2. Connect MODEL to **TensorRT Refit Loader** (`source_model`)
3. Select your engine file and model type
4. Run — refit takes seconds, generates at TRT speed
5. Change LoRA/strength/stack → rerun → seconds again

CLIP and text encoding still use the normal LoRA pipeline — only the UNet is TRT-accelerated.

## FAQ

### Is a refit-enabled engine slower than a normal one?

<!-- TODO: trt.BuilderFlag.REFIT prevents certain weight fusion/folding
     optimizations since weights must remain individually addressable -->

Slightly larger and marginally slower (single-digit percent). In practice the difference is small.

### Can I use a refit-enabled engine without refitting?

Yes. The regular `TensorRT Loader` runs it fine — it doesn't care about the refit flag. One engine works both ways: directly (base weights) or via `TensorRT Refit Loader` (with LoRA).

### Does adding/removing LoRAs work?

Yes. Stack multiple LoRAs, adjust strengths, remove some — the engine architecture is fixed but all weights are replaceable.
