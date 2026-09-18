import torch
import torch.nn.functional as F


# ==================================================
# DATA
# ==================================================

text = """
the cat sleeps on the mat.
the cat walks through the house.
the dog sleeps near the door.
the dog walks through the garden.
the cat sees the dog.
the dog sees the cat.
""".strip()


# ==================================================
# TOKENIZER
# ==================================================

vocabulary = sorted(set(text))
vocab_size = len(vocabulary)

stoi = {
    character: index
    for index, character in enumerate(vocabulary)
}

itos = {
    index: character
    for character, index in stoi.items()
}

tokens = torch.tensor(
    [stoi[c] for c in text],
    dtype=torch.long
)

print("Vocabulary size:", vocab_size)


# ==================================================
# TRAINING WINDOWS
# ==================================================

context_length = 8

X = []
Y = []


for i in range(
    len(tokens) - context_length
):

    # Example:
    #
    # X = "the cat "
    #
    # Y = "he cat s"
    #
    # Every position learns to predict
    # the following character.

    x = tokens[
        i:i + context_length
    ]

    y = tokens[
        i + 1:i + context_length + 1
    ]

    X.append(x)
    Y.append(y)


X = torch.stack(X)
Y = torch.stack(Y)


print("Training sequences:", len(X))
print("X shape:", X.shape)
print("Y shape:", Y.shape)


# ==================================================
# MODEL SIZE
# ==================================================

torch.manual_seed(42)

d_model = 16
head_size = 16


# ==================================================
# EMBEDDINGS
# ==================================================

token_embedding = (
    torch.randn(
        vocab_size,
        d_model
    ) * 0.1
).requires_grad_()


position_embedding = (
    torch.randn(
        context_length,
        d_model
    ) * 0.1
).requires_grad_()


# ==================================================
# ATTENTION PARAMETERS
# ==================================================

W_Q = (
    torch.randn(
        d_model,
        head_size
    ) * 0.1
).requires_grad_()


W_K = (
    torch.randn(
        d_model,
        head_size
    ) * 0.1
).requires_grad_()


W_V = (
    torch.randn(
        d_model,
        head_size
    ) * 0.1
).requires_grad_()


# ==================================================
# OUTPUT LAYER
# ==================================================

W_out = (
    torch.randn(
        head_size,
        vocab_size
    ) * 0.1
).requires_grad_()


b_out = torch.zeros(
    vocab_size,
    requires_grad=True
)


parameters = [
    token_embedding,
    position_embedding,
    W_Q,
    W_K,
    W_V,
    W_out,
    b_out
]


print(
    "Trainable parameters:",
    sum(p.numel() for p in parameters)
)


# ==================================================
# CAUSAL MASK
# ==================================================

mask = torch.triu(
    torch.ones(
        context_length,
        context_length,
        dtype=torch.bool
    ),
    diagonal=1
)


# ==================================================
# FORWARD PASS
# ==================================================

def forward(X):

    B, T = X.shape


    # ----------------------------------------------
    # TOKEN EMBEDDINGS
    # ----------------------------------------------

    tok = token_embedding[X]

    # Shape:
    # [B, T, d_model]


    # ----------------------------------------------
    # POSITION EMBEDDINGS
    # ----------------------------------------------

    pos = position_embedding[:T]

    # Broadcasting adds the same position vectors
    # to every sequence in the batch.

    x = tok + pos


    # ----------------------------------------------
    # CREATE Q, K, V
    # ----------------------------------------------

    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V


    # ----------------------------------------------
    # ATTENTION SCORES
    # ----------------------------------------------

    scores = (
        Q @ K.transpose(-2, -1)
    ) / (head_size ** 0.5)


    # Shape:
    # [B, T, T]


    # ----------------------------------------------
    # CAUSAL MASK
    # ----------------------------------------------

    current_mask = mask[:T, :T]

    scores = scores.masked_fill(
        current_mask,
        float("-inf")
    )


    # ----------------------------------------------
    # ATTENTION WEIGHTS
    # ----------------------------------------------

    attention = F.softmax(
        scores,
        dim=-1
    )


    # ----------------------------------------------
    # MIX VALUE VECTORS
    # ----------------------------------------------

    context = attention @ V

    # Shape:
    # [B, T, head_size]


    # ----------------------------------------------
    # PREDICT NEXT CHARACTER
    # ----------------------------------------------

    logits = context @ W_out + b_out

    # Shape:
    # [B, T, vocab_size]

    return logits, attention


# ==================================================
# TRAINING
# ==================================================

learning_rate = 0.1
epochs = 2000


for epoch in range(epochs):

    logits, attention = forward(X)


    loss = F.cross_entropy(
        logits.reshape(
            -1,
            vocab_size
        ),
        Y.reshape(-1)
    )


    # Clear previous gradients.
    for parameter in parameters:
        parameter.grad = None


    # Backpropagation.
    loss.backward()


    # Gradient descent.
    with torch.no_grad():

        for parameter in parameters:

            parameter -= (
                learning_rate
                * parameter.grad
            )


    if epoch % 200 == 0:

        print(
            f"epoch={epoch:4d} | "
            f"loss={loss.item():.6f}"
        )


# ==================================================
# GENERATION
# ==================================================

print("\nGENERATED TEXT:\n")


generated = "the "


for _ in range(300):

    context_string = generated[
        -context_length:
    ]


    x = torch.tensor(
        [[stoi[c] for c in context_string]],
        dtype=torch.long
    )


    with torch.no_grad():

        logits, _ = forward(x)


        # Only use prediction from the final
        # position in the context.
        final_logits = logits[0, -1]


        probabilities = F.softmax(
            final_logits,
            dim=-1
        )


        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()


    generated += itos[next_token]


print(generated)