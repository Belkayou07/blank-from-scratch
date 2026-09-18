# Blank From Scratch

A learning project where I build a small GPT-style language model from the basics.

The goal is not to copy a finished AI model. The goal is to understand the main ideas by building them step by step.

## Where the project is now

Current stage: **Blank-2**

- tokenizer: custom BPE, 4,096 tokens
- dataset v3: 3 billion characters
- tokenized data: 875,787,809 tokens
- Blank-2 size: 61,627,520 parameters
- context: 512 tokens
- architecture: decoder-only Transformer
- training stack: PyTorch + ROCm, BF16, SDPA, fused AdamW
- status: smoke test passed; full Blank-2 training not started yet

## The idea in one sentence

A language model reads earlier tokens and learns to predict the next token.

## Repository map

- `01-foundations/` — first models, gradients, small neural networks
- `02-language-models/` — bigram, context and embedding models
- `03-attention/` — self-attention and Transformer blocks
- `04-mini-gpt/` — first real small GPT-style models
- `05-tokenizer/` — BPE tokenizer work
- `06-data/` — dataset building and tokenization scripts
- `07-blank-0/` — first larger general model
- `08-blank-1/` — 33.56M parameter model
- `09-blank-2/` — current 61.63M parameter model
- `benchmarks/` — GPU and speed tests
- `docs/` — short explanations and results

The large datasets, checkpoints and virtual environments are intentionally not stored in Git.

See [JOURNEY.md](JOURNEY.md) for the full path in simple language.
