# Journey

1. **First model** — learned weights, bias, loss and gradient descent.
2. **Autograd** — built a tiny system that calculates gradients automatically.
3. **Small neural network** — learned why nonlinear layers matter.
4. **Bigram language model** — predicted the next character from one previous character.
5. **More context** — used several previous characters.
6. **Embeddings** — turned token IDs into learned vectors.
7. **Attention** — tokens learned to use information from earlier tokens.
8. **Transformer** — combined attention, MLPs, normalization and residual connections.
9. **Mini GPT** — trained a real small decoder-only Transformer on Tiny Shakespeare.
10. **BPE tokenizer** — changed text into reusable subword/byte tokens.
11. **Real data** — moved to FineWeb-Edu and scaled from 100M to 3B characters.
12. **Blank-0** — first larger general-language model, about 15.7M parameters.
13. **Blank-1** — 33.56M parameters, best validation loss 2.9193.
14. **Blank-2** — 61.63M parameters; current stage.

Main pipeline:

`text -> tokens -> embeddings -> Transformer -> next-token probabilities -> loss -> gradients -> parameter update`

A base model is not automatically a chatbot. Instruction tuning comes later.
