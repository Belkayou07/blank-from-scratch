# Results So Far

| Model | Parameters | Data | Best validation loss |
|---|---:|---:|---:|
| Blank-0 v0.1 | 15.74M | ~29M tokens | ~3.45 |
| Blank-0 v0.2 | 15.74M | 285.7M tokens | 3.0687 |
| Blank-1 v0.1 | 33.56M | 285.7M tokens | 2.9193 |
| Blank-2 | 61.63M | 858.3M training tokens available | not fully trained |

Blank-2 smoke test: about 75k tokens/s at batch 16; BF16, SDPA and fused AdamW were stable.
