# Roadmap

Current point: **Blank-2 smoke test passed.**

## Next

1. Decide the final Blank-2 training setup.
2. Train Blank-2.
3. Compare Blank-0, Blank-1 and Blank-2.
4. Improve the training data before blindly making the model bigger.

## Later

The long-term path is:

```text
better base model
    ↓
better / broader data
    ↓
instruction tuning
    ↓
assistant-style conversations
    ↓
evaluation
    ↓
local chat app
    ↓
RAG / tools / memory
```

## Final goal

Not a frontier model.

The goal is a small, useful model that I understand from:

```text
text
→ tokens
→ embeddings
→ Transformer
→ prediction
→ loss
→ gradients
→ training
```

Then I can build assistant features around it.
