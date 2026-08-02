# TensorRT Nodes UX — TODO

## Current State

- **TensorRT Loader** — loads any engine (including refit-enabled), runs with base weights only. Manual engine selection dropdown.
- **TensorRT Refit Loader** — loads refit-enabled engine + source MODEL, refits LoRA weights in. Manual engine selection dropdown.
- Both require a separate build step (Static/Dynamic TRT Conversion nodes) before first use.

## Proposed: TensorRT Loader Auto

Single node that handles build, load, and refit automatically based on the input MODEL.

### Inputs

- **MODEL** (required) — from checkpoint loader, possibly with LoRAs applied
- **model_type** — sdxl, sd1x, sd2x, etc.

### Widgets

#### Always visible
- **filename_prefix** (string) — supports `{modelname}` and `{lora_hash}` placeholders. User is responsible for meaningful naming since LoRA config is not auto-encoded into the filename. Tooltip must explain this clearly.
- **refit** (bool) — build engine with REFIT flag, refit LoRA weights at load time
- **build_if_absent** (bool) — when enabled, unhides all build-related widgets. When disabled, only loads existing engines (errors if not found).

#### Visible when `build_if_absent` is enabled
- **static_shapes** (bool) — toggles between static and dynamic mode, controls which resolution/batch widgets are visible
- **context_len** (int) — max CLIP token count (default 308 for SDXL with prompt weighting)
- **enable_refit** (bool) — build with REFIT flag (should default to match the `refit` widget above)

**Static mode widgets** (visible when `static_shapes` is true):
- **height** / **width** / **batch_size**

**Dynamic mode widgets** (visible when `static_shapes` is false):
- **min_height** / **opt_height** / **max_height**
- **min_width** / **opt_width** / **max_width**
- **min_batch** / **opt_batch** / **max_batch**

#### Disk management widgets
- **disk_management** (bool) — when enabled, unhides `max_disk_usage_gb`
- **max_disk_usage_gb** (float) — auto-evict oldest engines (FIFO) from the managed directory when usage exceeds this limit

### Widget Visibility (JS)

Three levels of toggle, all implemented in companion JS file (like existing `builder.js`):

1. **`build_if_absent`** — shows/hides ALL build widgets (static_shapes, context_len, enable_refit, resolution, batch, and the static/dynamic sub-widgets)
2. **`static_shapes`** (within build widgets) — shows static widgets OR dynamic widgets, never both
3. **`disk_management`** — shows/hides `max_disk_usage_gb`

### Behavior

1. **Check for existing engine** matching `filename_prefix` + profile description on disk
2. **Decision tree:**
   - Engine exists on disk?
     - Yes + no LoRAs (or refit disabled) → load and run
     - Yes + LoRAs + refit enabled → load, refit with LoRA weights, run
     - No + `build_if_absent` enabled → build engine, save to disk, then load (+ refit if LoRAs)
     - No + `build_if_absent` disabled → error
3. **Disk management** (when enabled):
   - Engines go to `models/tensorrt/auto_managed/`
   - Before building a new engine, check total directory size
   - If current + estimated new engine size > `max_disk_usage_gb`, delete oldest engines (by mtime, FIFO) until there's room
   - Never delete an engine that's currently loaded

### Engine Naming & Identity

#### Filename strategy

Engine filenames are **user-controlled** via `filename_prefix`. LoRA configuration is deliberately NOT auto-encoded into the filename — the engine is built from the base model, and LoRAs are refitted in at load time.

```
models/tensorrt/auto_managed/
  {filename_prefix}_{profile_desc}.engine
  {filename_prefix}_{profile_desc}.weight_map.json
```

Where `profile_desc` is like `stat-b1-h1024-w1024` (existing convention).

**Available placeholders in `filename_prefix`:**
- `{modelname}` — resolved from upstream checkpoint loader via workflow graph trace
- `{lora_hash}` — short hash of the LoRA stack (sorted lora names + strengths). Useful if the user wants separate engines per LoRA combo, though this defeats the purpose of refit.

**Tooltip text (important):** "Engine filename prefix. The engine is built from the base model weights — LoRAs are applied via refit at load time, not baked into the engine. Use {modelname} to auto-insert the checkpoint name. Use {lora_hash} only if you want separate engines per LoRA combination (unusual)."

#### Refit identity (within-session)

To avoid redundant 13s refits on consecutive runs with the same LoRA config:
- Cache `patches_uuid` from the last successful refit
- If `patches_uuid` hasn't changed → skip refit, reuse VRAM engine
- `patches_uuid` is non-deterministic (new each session) but stable within a session

#### What ModelPatcher provides (and doesn't)

ComfyUI's `ModelPatcher` has **no content hash and no source filename**. What it does have:

- **`patches_uuid`** — a UUID4 regenerated on every patch add/remove. Cheap identity check but non-deterministic across sessions. Useful for within-session refit caching.
- **`self.patches`** dict — maps weight keys → `(strength, patch_data, strength_model, offset, function)` tuples. Enumerates exactly which weights are patched and at what strengths, but **LoRA filenames are not stored** — only the weight deltas.
- **`self.model`** — the underlying `nn.Module`. No source path, no checkpoint name.
- **`model_size()`** — byte count, not a fingerprint.

**Graph trace** (via hidden `PROMPT`/`UNIQUE_ID`) resolves `{modelname}` by finding the upstream checkpoint loader. This is a single-hop trace (proven in VAE builder). We do NOT trace LoRA loaders — that path is fragile with arbitrary chain depths and third-party loader variants.

### Open Questions

- Dynamic engines: one engine covers a resolution range, so the filename shouldn't include specific resolution — just the profile bounds. How to encode this cleanly?
- Should there be a "force rebuild" button/option?
- How to handle the case where `refit` is enabled but the existing engine on disk was built without `enable_refit`? Error with clear message? Auto-rebuild?
