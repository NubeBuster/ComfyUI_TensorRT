"""Multi-slot deterministic refit cache for TRT LoRA engines.

Instead of keying on ComfyUI's random patches_uuid (which regenerates on
every LoRA application, making A→B→A always re-refit), we compute a
deterministic hash from the actual LoRA patch content: which keys are
patched, at what strengths, with what tensor shapes and checksums.

This lets us cache multiple refitted engines on disk and skip refitting
when cycling between known LoRA configurations.
"""

import hashlib
import logging
import os

log = logging.getLogger("comfyui_tensorrt")


def compute_patches_hash(model):
    """Compute a deterministic hash of a model's LoRA patches.

    Returns a short hex string (12 chars) that uniquely identifies the
    combination of LoRA patches applied to the model. Returns None if
    the model has no patches.

    The hash is based on: sorted patch key names, strength values,
    tensor shapes, and raw tensor bytes.
    """
    if not hasattr(model, "patches") or not model.patches:
        return None

    h = hashlib.sha256()

    for key in sorted(model.patches.keys()):
        h.update(key.encode())
        for patch_tuple in model.patches[key]:
            # patch_tuple: (strength_patch, patch_data, strength_model, offset, function)
            strength_patch = patch_tuple[0]
            patch_data = patch_tuple[1]
            strength_model = patch_tuple[2]

            # Hash strengths
            h.update(f"{strength_patch:.8f}:{strength_model:.8f}".encode())

            # Hash tensor fingerprint from patch_data
            _hash_patch_data(h, patch_data)

    result = h.hexdigest()[:12]
    log.info("patches_hash: %d keys, result=%s", len(model.patches), result)
    return result


def _hash_patch_data(h, data):
    """Hash a patch data element (tensor, tuple, or dict)."""
    import torch

    if isinstance(data, torch.Tensor):
        # Shape + deterministic point samples (fast, no float accumulation)
        h.update(str(tuple(data.shape)).encode())
        flat = data.flatten()
        n = flat.numel()
        if n > 0:
            # Sample first, last, mid, and a few stride points
            indices = [0, n - 1, n // 2, n // 4, 3 * n // 4]
            for i in indices:
                if i < n:
                    h.update(f"{flat[i].item():.8f}".encode())
    elif hasattr(data, "weights"):
        # WeightAdapterBase subclasses (LoRAAdapter, GLoRAAdapter, etc.)
        # have a .weights tuple of tensors/scalars — hash that instead
        # of str(obj) which includes non-deterministic memory addresses.
        h.update(type(data).__name__.encode())
        _hash_patch_data(h, data.weights)
    elif isinstance(data, (tuple, list)):
        for item in data:
            _hash_patch_data(h, item)
    elif isinstance(data, dict):
        for k in sorted(data.keys()):
            h.update(str(k).encode())
            _hash_patch_data(h, data[k])
    elif isinstance(data, (int, float)):
        h.update(f"{data:.8f}".encode())
    elif data is not None:
        h.update(str(data).encode())


# ---------------------------------------------------------------------------
# In-memory multi-slot cache
# ---------------------------------------------------------------------------

# {(engine_path, lora_hash): {"patcher": ..., "lru_tick": int}}
_mem_cache = {}
_lru_counter = 0


def mem_lookup(engine_path, lora_hash):
    """Look up a cached patcher by engine path and LoRA hash.

    Returns the patcher if found and its engine or disk file is valid,
    else None. Updates LRU tick on hit.
    """
    global _lru_counter
    key = (engine_path, lora_hash)
    entry = _mem_cache.get(key)
    if entry is None:
        return None

    patcher = entry["patcher"]
    if patcher is None:
        return None

    cached_unet = patcher.model.diffusion_model

    # Engine still in VRAM
    if cached_unet.engine is not None:
        _lru_counter += 1
        entry["lru_tick"] = _lru_counter
        log.info("Refit cache: memory hit (engine in VRAM) hash=%s", lora_hash)
        return patcher

    # Engine evicted but persisted file exists on disk
    if cached_unet.engine_path and os.path.isfile(cached_unet.engine_path):
        _lru_counter += 1
        entry["lru_tick"] = _lru_counter
        log.info(
            "Refit cache: memory hit (evicted, will reload from %s) hash=%s",
            cached_unet.engine_path,
            lora_hash,
        )
        return patcher

    # Invalid entry — persisted file gone
    log.info("Refit cache: memory entry invalid (disk file missing) hash=%s", lora_hash)
    del _mem_cache[key]
    return None


def mem_store(engine_path, lora_hash, patcher):
    """Store a patcher in the multi-slot memory cache."""
    global _lru_counter
    _lru_counter += 1
    _mem_cache[(engine_path, lora_hash)] = {
        "patcher": patcher,
        "lru_tick": _lru_counter,
    }
    log.info("Refit cache: stored hash=%s (slots=%d)", lora_hash, len(_mem_cache))


def mem_release_engine(engine_path, lora_hash):
    """Release a specific cache entry's engine (called on cache miss before refit)."""
    key = (engine_path, lora_hash)
    entry = _mem_cache.get(key)
    if entry and entry["patcher"] is not None:
        entry["patcher"].model.diffusion_model._unload()
        entry["patcher"] = None


def mem_release_all_engines():
    """Release all cached TRT engines from VRAM. Called before loading a new engine."""
    for key, entry in list(_mem_cache.items()):
        if entry["patcher"] is not None:
            unet = entry["patcher"].model.diffusion_model
            if unet.engine is not None:
                log.info(
                    "Refit cache: releasing engine for hash=%s path=%s",
                    key[1],
                    unet.engine_path,
                )
                unet._unload()


def mem_clear():
    """Clear the entire memory cache (e.g., on force unload)."""
    for key, entry in list(_mem_cache.items()):
        if entry["patcher"] is not None:
            entry["patcher"].model.diffusion_model._unload()
    _mem_cache.clear()
    log.info("Refit cache: memory cache cleared")


# ---------------------------------------------------------------------------
# Disk cache paths
# ---------------------------------------------------------------------------


def disk_cache_path(engine_path, lora_hash):
    """Return the disk path for a refitted engine with a specific LoRA hash.

    Format: <engine_dir>/.refit_cache/<engine_stem>_<hash>.engine
    """
    engine_dir = os.path.dirname(engine_path)
    cache_dir = os.path.join(engine_dir, ".refit_cache")
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(engine_path))[0]
    return os.path.join(cache_dir, f"{stem}_{lora_hash}.engine")


def disk_lookup(engine_path, lora_hash):
    """Check if a refitted engine exists on disk for this hash.

    Returns the path if it exists, else None.
    """
    path = disk_cache_path(engine_path, lora_hash)
    if os.path.isfile(path):
        log.info("Refit cache: disk hit hash=%s path=%s", lora_hash, path)
        return path
    return None
