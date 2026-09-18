# Blank From Scratch

A learning project where I build a small GPT-style language model from the basics.

The goal is not to copy a finished AI model. The goal is to understand the main ideas by building them step by step.

## Current stage

**Blank-2**

- custom BPE tokenizer: 4,096 tokens
- Dataset v3: 3 billion characters
- tokenized data: 875,787,809 tokens
- training split: 858,272,052 tokens
- model size: 61,627,520 parameters
- context: 512 tokens
- architecture: decoder-only Transformer
- stack: PyTorch + ROCm, BF16, SDPA, fused AdamW
- status: smoke test passed; full Blank-2 training has not started

## The main idea

```text
previous tokens → model → next-token prediction
                         ↓
                        loss
                         ↓
                      gradients
                         ↓
                 update parameters
```

## Repository

- `01-foundations/` — weight, bias, gradients, autograd, MLP
- `02-language-models/` — bigram, context and embeddings
- `03-attention/` — self-attention and Transformers
- `04-mini-gpt/` — first GPT-style models
- `05-tokenizer/` — BPE tokenizer
- `06-data/` — dataset builders, tokenization and metadata
- `07-blank-0/` — first larger base model
- `08-blank-1/` — 33.56M parameter model
- `09-blank-2/` — current 61.63M parameter model
- `benchmarks/` — GPU and speed tests
- `docs/` — short explanations, results and roadmap

## Read these first

- [JOURNEY.md](JOURNEY.md) — the whole journey in simple language
- [Core ideas](docs/CORE_IDEAS.md) — short explanations
- [Results](docs/RESULTS.md) — what each scale-up changed
- [Setup](docs/SETUP.md) — packages and large-file notes
- [Roadmap](docs/ROADMAP.md) — what comes next

## Large files

Datasets, token binaries, checkpoints, virtual environments and caches are not committed.

They are too large for a normal Git repository and can be rebuilt from the scripts.

Some of the long training scripts were formatting-cleaned when this learning folder was reorganized. The model logic and experiment settings were kept, while large generated artifacts were left out.
