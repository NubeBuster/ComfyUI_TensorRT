# LoRA Refit

<details>
  <summary>
    SDXL Refit (apply LoRAs to a TRT engine in ~30s not ~5-10m) implemented and working for me.
  </summary>
Implemented, tested, fixed bug, repeat 5 times? Now it works mostly.
I've fully focused the debugging on having it work for XY Plotting. In XY plot you don't want to unload the engine between each iteration, or the TRT VAE model..? In short, as far as I know and have tested:
- Executing an XY Plot (from EasyUse) would cause model load/unload overhead in my scenario (providing a TRT VAE + Unet REFIT model), the comfyui API allegedly doesn't properly support this niche case where we do not want to unload the TRT VAE nor the Unit model between XY Plot inference iterations on memory pressure, but specifically only during the XY Plot  Execution: behave normally for other model eviction events, without modifying the external custom nodes like EasyUse.
  </summary>
</details>

# SDXL VAE TRT

<details>
  <summary>Benchmark report I mentioned earlier in this issue benchmark was found to be invalid, effectiveness TBD.</summary>

I ran one test on a given XY Plot SDXL T2I workflow on fresh ComfyUI restart assuring the refitted model cache on disk is present.
**Model**: [BigLove_Pony3](https://civitai.com/models/897413?modelVersionId=1973705) with a bunch or LoRAs applied that I left unchanged between test runs.
**XY-Plot**:
| # | X-Sampler | X-Scheduler | Y-Seeds
|---|---------|-----------| --------- |
| 1 | dpmpp_2m_sde_gpu | beta | 8 |
| 2 | dpmpp_2m | beta |8 |
| 3 | euler | beta |8 |
| 4 | euler_ancestral | karras |8 |

**Total**: 32 XY-Plot Iterations (i.e. Images)

**Results**:

- Prompt executed in 335.11 seconds
- Prompt executed in 334.05 seconds

**Conclusion**:

<img width="318" height="159" alt="Image" src="https://github.com/user-attachments/assets/55b34095-682a-4181-88c7-f3fcaffaa203" />

... ah wait [dynamic vram](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.16.0) - Mar 5th (moreover, new, maybe my VAE TRT Implementation became obsolete, and I haven't failed to properly test it in the past)

Lemme set `--disable-dynamic-vram`...
VAE No-TRT: 347.61 seconds
VAE TRT: 377.42 seconds

<img width="318" height="159" alt="Image" src="https://github.com/user-attachments/assets/55b34095-682a-4181-88c7-f3fcaffaa203" />

After some hard self-reflection I retraced my steps back to the workflow of the original VAE test, and I was able to reproduce the results, and then not... inconsistent. Then a new fact came up:

> TRT engine's first inference after loading incurs a one-time warm-up cost (CUDA kernel selection + scratch memory allocation) that doesn't exist for normal PyTorch models, so any benchmark must discard the first run from both to compare steady-state performance.

Benchmark but discarding first result, reduced test to sampler_scheduler_pairs\*2seeds=8:

- VAE TRT: Prompt executed in 86.52 seconds
- No VAE TRT: Prompt executed in 85.37 seconds

![Questioning sanity](https://media1.tenor.com/m/cFsekjlQb1wAAAAd/quantum-leap-mirror.gif)

Possible causes:

- SDXL VAE is not a significant enough compute to compensate for the TRT init overhead
- The XY Plot workflows I am testing in are the worst case scenario for VAE TRT (I reckon TRT shows more benefit on batch>1 compared to batch=1,repeat that XY Plot does)
- ComfyUI's implementations have evolved since back then and the speedup is no longer feasible

</details>

At least the SDXL LoRA refitting is a success. Applying LoRAs to an already built SDXL engine (inpaint also supported in my fork), now takes 30-35s on an RTX 4060TI 16GB, compared to having to rebuild the engine, which takes 5m+. I have been able to swap LoRAs in my SDXL workflows with comparatively very little notable overhead. For most workflow runs the speed gain of TRT outweighs the refit overhead at the first execution.
