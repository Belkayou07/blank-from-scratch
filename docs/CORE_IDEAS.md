# Core Ideas

## Token

A token is a small piece of text represented by an ID.

Example:

`"computer" → tokens → numbers`

The model works with numbers, not raw words.

## Embedding

An embedding changes a token ID into a learned vector.

The vector is the model's internal representation of that token.

## Parameter

A parameter is a number inside the model that training can change.

A model with 60M parameters has about 60 million trainable numbers.

## Training

The model sees text and predicts the next token.

We compare the prediction with the real next token.

The difference becomes the **loss**.

Training changes the parameters to reduce that loss.

## Gradient

A gradient tells the optimizer how a parameter affects the loss.

It helps decide which way the parameter should move.

## Attention

Attention lets one token use information from earlier tokens.

Query asks what information is useful.

Key describes what information a token has.

Value carries the information.

## Transformer

A Transformer block mainly contains:

- attention
- an MLP
- normalization
- residual connections

Many blocks are stacked together.

## Context window

The context window is how many earlier tokens the model can directly see.

Blank-2 currently uses 512 tokens.

With the current tokenizer, that is roughly 1,750 characters on average.

## Pretraining

Pretraining teaches the model general language by next-token prediction.

It does not automatically teach the model to act like an assistant.

## Instruction tuning

Instruction tuning later teaches patterns such as:

`User: question`

`Assistant: answer`

That is one of the steps needed before Blank behaves like a chatbot.
