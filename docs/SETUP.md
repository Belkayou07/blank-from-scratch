# Setup

This repo is mainly a record of the learning journey.

## Basic Python packages

Install:

```powershell
pip install numpy tokenizers datasets
```

PyTorch + ROCm depends on the GPU and OS, so it is not pinned here.

The experiments were run on an AMD GPU with a ROCm-enabled PyTorch build.

## Large files

These are not stored in Git:

- training datasets
- token binaries
- model checkpoints
- virtual environments
- caches

They can become several GB.

Use the scripts in `06-data/` to rebuild the datasets and token files.

The tokenizer JSON is also generated locally by `05-tokenizer/train_tokenizer_v1.py`.

## Important

This is an educational repo, not a polished ML library.

The scripts show the project in the order it was learned.
