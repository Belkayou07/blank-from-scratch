# Results So Far

## Main model progression

| Model | Parameters | Training data | Best validation loss |
|---|---:|---:|---:|
| Blank-0 v0.1 | 15.74M | ~29M tokens | ~3.45 |
| Blank-0 v0.2 | 15.74M | 285.7M tokens | 3.0687 |
| Blank-1 v0.1 | 33.56M | 285.7M tokens | 2.9193 |
| Blank-2 | 61.63M | 858.3M training tokens available | not fully trained yet |

## What the experiments showed

### More data helped

Blank-0 stayed the same size, but moving from about 29M tokens to 285.7M tokens reduced validation loss strongly.

### More model capacity helped

Blank-1 used the same Dataset v2 as Blank-0 v0.2 but had more parameters.

Validation loss improved from about 3.0687 to 2.9193.

### Faster math mattered a lot

The project moved from slower FP32/manual attention to:

- BF16
- PyTorch SDPA
- fused AdamW

This made training much faster.

### Blank-1 was still not a chatbot

Its text looked more natural, but it could:

- repeat itself
- lose the topic
- invent facts
- produce confident nonsense

That is expected from a small base model trained only for next-token prediction.

## Dataset v3

- 3,000,000,000 characters
- 875,787,809 total tokens
- 858,272,052 training tokens
- 17,515,757 validation tokens
- 4,096-token BPE vocabulary
- about 3.4255 characters per token

## Blank-2 smoke test

- 61,627,520 parameters
- 12 layers
- width 640
- 10 attention heads
- context 512
- BF16 + SDPA + fused AdamW
- best tested batch: 16
- about 75,001 tokens/second
- final smoke validation loss after 300 updates: 5.0843
- training health checks passed

This smoke result only proves the setup trains correctly. It is not the final Blank-2 result.
