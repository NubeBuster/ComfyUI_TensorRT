"""Benchmark: TRT VAE decode vs normal VAE decode.

Run inside the ComfyUI container:
    cd /app && python custom_nodes/comfyui_tensorrt/tests/bench_vae_decode.py

Decodes a random latent N times with each VAE, discards first run (warm-up),
reports per-decode timings.
"""

import argparse
import os
import sys
import time

# Bootstrap ComfyUI imports (run from /app)
sys.path.insert(0, "/app")
os.chdir("/app")

# Prevent ComfyUI server from starting — stub PromptServer before anything imports it
import types
server_mod = types.ModuleType("server")
class _FakePS:
    instance = None
    @staticmethod
    def send_sync(*a, **kw): pass
server_mod.PromptServer = _FakePS
sys.modules["server"] = server_mod

import torch
import folder_paths


def load_normal_vae(ckpt_name):
    """Load VAE from a checkpoint."""
    from nodes import CheckpointLoaderSimple

    loader = CheckpointLoaderSimple()
    _, _, vae = loader.load_checkpoint(ckpt_name)
    return vae


def load_trt_vae(decode_engine):
    """Load TRT VAE directly, bypassing node discovery."""
    import importlib
    importlib.import_module("custom_nodes.comfyui_tensorrt")
    loader_mod = importlib.import_module("custom_nodes.comfyui_tensorrt.tensorrt_loader")

    dec_path = os.path.join(folder_paths.models_dir, "tensorrt", decode_engine)
    if not os.path.isfile(dec_path):
        raise FileNotFoundError(f"Decode engine not found: {dec_path}")
    return loader_mod.TrtVAE(dec_path, None)


def bench_decode(vae, latent, n_runs, label):
    """Decode n_runs times, return list of durations (skipping first)."""
    times = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        vae.decode(latent["samples"])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dur = t1 - t0
        tag = " (warm-up, discarded)" if i == 0 else ""
        print(f"  [{label}] run {i}: {dur:.4f}s{tag}")
        if i > 0:
            times.append(dur)
    return times


def main():
    parser = argparse.ArgumentParser(description="Benchmark VAE decode: TRT vs normal")
    parser.add_argument("--ckpt", default="Pony/bigLove_pony3.safetensors", help="Checkpoint name for normal VAE")
    parser.add_argument("--decode-engine", default="vae/VAE_STAT_bigLove_pony3_decode_$stat-h-1216-w-832_00001_.engine", help="TRT decode engine path relative to tensorrt dir")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=1216)
    parser.add_argument("--runs", type=int, default=10, help="Number of decode runs (first is warm-up)")
    args = parser.parse_args()

    latent_h, latent_w = args.height // 8, args.width // 8
    latent = {"samples": torch.randn(1, 4, latent_h, latent_w, device="cuda", dtype=torch.float32)}

    print(f"\nLatent: {latent['samples'].shape} ({args.width}x{args.height})")
    print(f"Runs: {args.runs} (first discarded as warm-up)\n")

    # --- Normal VAE ---
    print("Loading normal VAE...")
    normal_vae = load_normal_vae(args.ckpt)
    normal_times = bench_decode(normal_vae, latent, args.runs, "Normal")
    del normal_vae
    torch.cuda.empty_cache()

    # --- TRT VAE ---
    print("\nLoading TRT VAE...")
    trt_vae = load_trt_vae(args.decode_engine)
    trt_times = bench_decode(trt_vae, latent, args.runs, "TRT")
    del trt_vae
    torch.cuda.empty_cache()

    # --- Summary ---
    normal_avg = sum(normal_times) / len(normal_times)
    trt_avg = sum(trt_times) / len(trt_times)
    speedup = normal_avg / trt_avg

    print(f"\n{'='*50}")
    print(f"Normal VAE avg: {normal_avg:.4f}s")
    print(f"TRT VAE avg:    {trt_avg:.4f}s")
    print(f"Speedup:        {speedup:.2f}x")
    if speedup < 1.05:
        print("Verdict:        No meaningful speedup")
    elif speedup < 1.5:
        print("Verdict:        Marginal speedup")
    else:
        print(f"Verdict:        {speedup:.1f}x faster")


if __name__ == "__main__":
    main()
