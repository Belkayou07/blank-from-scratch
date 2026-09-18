# Blank From Scratch

A learning project where I build a small GPT-style language model from the basics.

The goal is to understand the main ideas step by step, from one weight and bias to a decoder-only Transformer.

## Current stage

**Blank-2**

- 61,627,520 parameters
- 12 layers
- width 640
- 10 attention heads
- context 512
- custom 4,096-token BPE tokenizer
- Dataset v3: 3 billion characters / 875,787,809 tokens
- BF16 + SDPA + fused AdamW
- smoke test passed; full Blank-2 pretraining not started yet

See `JOURNEY.md` for the short learning path.

Large datasets, virtual environments and checkpoints are intentionally not stored in Git.
