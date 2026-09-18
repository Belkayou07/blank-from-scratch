# Journey

This file is a short record of what I learned while building the model.

## 1. First model

I started with one simple formula:

`prediction = weight × input + bias`

I learned that a model has numbers called **parameters**. Training changes these numbers so the error becomes smaller.

## 2. Gradients

I built a tiny automatic-gradient system.

A gradient tells us which direction a parameter should move to reduce the loss.

## 3. Small neural network

I built a small MLP.

This showed that several neurons with nonlinear functions can learn patterns that one straight line cannot.

## 4. Bigram language model

The first language model only looked at one previous character.

It learned:

`current character → likely next character`

This was very limited, but it showed the basic idea of language modeling.

## 5. More context

I made models that looked at several previous characters.

More context gave the model more information, but exact lookup methods became sparse.

## 6. Embeddings

Tokens were changed into learned vectors.

Instead of treating every token as an isolated number, the model could learn useful internal representations.

## 7. Attention

I built self-attention with Query, Key and Value vectors.

Attention lets each token look at useful earlier tokens.

## 8. Transformer

I combined attention, an MLP, residual connections and normalization.

This became the first real Transformer block.

## 9. Mini GPT

I stacked Transformer blocks and trained on Tiny Shakespeare.

The model started to produce text that looked like Shakespeare, even when the text was not truly meaningful.

## 10. BPE tokenizer

I moved from single characters to byte-pair encoding.

Common groups of bytes can become one token.

This made text much shorter in token form.

## 11. Real dataset

I moved to FineWeb-Edu data.

The dataset grew from about 100M characters to 1B, and then to 3B characters.

## 12. Blank-0

Blank-0 had about **15.7M parameters**.

With more data, its validation loss improved a lot.

This showed that model size is not enough. Data matters too.

## 13. Blank-1

Blank-1 had **33.56M parameters**.

Best validation loss: **2.9193**.

It produced more natural English than Blank-0, but it still repeated itself, drifted off topic and invented facts.

## 14. Blank-2

Blank-2 has **61.63M parameters**.

Dataset v3 has **858.27M training tokens**.

The smoke test passed:

- BF16 stable
- gradients stable
- fused AdamW working
- SDPA working
- best tested batch: 16
- measured speed: about 75k tokens/second

The full Blank-2 pretraining run has not started yet.

## What I understand now

The main pipeline is:

`text → tokens → embeddings → Transformer → next-token probabilities → loss → gradients → parameter update`

A base model is not automatically a chatbot.

Later, instruction tuning is needed to teach it how to answer users directly.
