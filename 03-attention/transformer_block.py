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
    character: i
    for i, character in enumerate(vocabulary)
}

itos = {
    i: character
    for character, i in stoi.items()
}


tokens = torch.tensor(
    [stoi[c] for c in text],
    dtype=torch.long
)


# ==================================================
# TRAINING WINDOWS
# ==================================================

context_length = 8

X = []
Y = []

for i in range(len(tokens) - context_length):

    X.append(
        tokens[i:i + context_length]
    )

    Y.append(
        tokens[i + 1:i + context_length + 1]
    )


X = torch.stack(X)
Y = torch.stack(Y)


print("Vocabulary size:", vocab_size)
print("Training sequences:", len(X))


# ==================================================
# MODEL CONFIGURATION
# ==================================================

torch.manual_seed(42)

d_model = 16
head_size = 16
mlp_hidden = 64


# ==================================================
# PARAMETERS
# ==================================================

def parameter(*shape, scale=0.1):

    return (
        torch.randn(*shape) * scale
    ).requires_grad_()


# Token + position embeddings

token_embedding = parameter(
    vocab_size,
    d_model
)

position_embedding = parameter(
    context_length,
    d_model
)


# Attention

W_Q = parameter(d_model, head_size)
W_K = parameter(d_model, head_size)
W_V = parameter(d_model, head_size)

W_attention_out = parameter(
    head_size,
    d_model
)


# First LayerNorm

ln1_gain = torch.ones(
    d_model,
    requires_grad=True
)

ln1_bias = torch.zeros(
    d_model,
    requires_grad=True
)


# Feed-forward MLP

W1 = parameter(
    d_model,
    mlp_hidden
)

b1 = torch.zeros(
    mlp_hidden,
    requires_grad=True
)

W2 = parameter(
    mlp_hidden,
    d_model
)

b2 = torch.zeros(
    d_model,
    requires_grad=True
)


# Second LayerNorm

ln2_gain = torch.ones(
    d_model,
    requires_grad=True
)

ln2_bias = torch.zeros(
    d_model,
    requires_grad=True
)


# Vocabulary output

W_vocab = parameter(
    d_model,
    vocab_size
)

b_vocab = torch.zeros(
    vocab_size,
    requires_grad=True
)


parameters = [
    token_embedding,
    position_embedding,

    W_Q,
    W_K,
    W_V,
    W_attention_out,

    ln1_gain,
    ln1_bias,

    W1,
    b1,
    W2,
    b2,

    ln2_gain,
    ln2_bias,

    W_vocab,
    b_vocab
]


print(
    "Trainable parameters:",
    sum(p.numel() for p in parameters)
)


# ==================================================
# CAUSAL MASK
# ==================================================

causal_mask = torch.triu(
    torch.ones(
        context_length,
        context_length,
        dtype=torch.bool
    ),
    diagonal=1
)


# ==================================================
# OUR OWN LAYERNORM
# ==================================================

def layer_norm(x, gain, bias):

    mean = x.mean(
        dim=-1,
        keepdim=True
    )

    variance = (
        (x - mean) ** 2
    ).mean(
        dim=-1,
        keepdim=True
    )

    normalized = (
        (x - mean)
        / torch.sqrt(variance + 1e-5)
    )

    return (
        gain * normalized
        + bias
    )


# ==================================================
# FORWARD
# ==================================================

def forward(X):

    B, T = X.shape


    # ----------------------------------------------
    # EMBEDDINGS
    # ----------------------------------------------

    x = (
        token_embedding[X]
        +
        position_embedding[:T]
    )


    # ==============================================
    # SELF-ATTENTION
    # ==============================================

    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V


    scores = (
        Q @ K.transpose(-2, -1)
    ) / (head_size ** 0.5)


    mask = causal_mask[:T, :T]

    scores = scores.masked_fill(
        mask,
        float("-inf")
    )


    attention_weights = F.softmax(
        scores,
        dim=-1
    )


    attention_output = (
        attention_weights @ V
    )


    attention_output = (
        attention_output
        @ W_attention_out
    )


    # ----------------------------------------------
    # RESIDUAL CONNECTION
    # ----------------------------------------------

    x = x + attention_output


    # ----------------------------------------------
    # LAYER NORMALIZATION
    # ----------------------------------------------

    x = layer_norm(
        x,
        ln1_gain,
        ln1_bias
    )


    # ==============================================
    # FEED-FORWARD NETWORK
    # ==============================================

    mlp = x @ W1 + b1

    mlp = F.relu(mlp)

    mlp = mlp @ W2 + b2


    # ----------------------------------------------
    # SECOND RESIDUAL
    # ----------------------------------------------

    x = x + mlp


    # ----------------------------------------------
    # SECOND LAYERNORM
    # ----------------------------------------------

    x = layer_norm(
        x,
        ln2_gain,
        ln2_bias
    )


    # ==============================================
    # VOCABULARY PREDICTION
    # ==============================================

    logits = (
        x @ W_vocab
        + b_vocab
    )


    return logits, attention_weights


# ==================================================
# TRAINING
# ==================================================

learning_rate = 0.05
epochs = 2500


for epoch in range(epochs):

    logits, attention = forward(X)


    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        Y.reshape(-1)
    )


    # Reset gradients
    for p in parameters:
        p.grad = None


    # Backpropagation
    loss.backward()


    # Gradient descent
    with torch.no_grad():

        for p in parameters:

            p -= (
                learning_rate
                * p.grad
            )


    if epoch % 250 == 0:

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

        logits, attention = forward(x)

        final_logits = logits[0, -1]

        probabilities = F.softmax(
            final_logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            1
        ).item()


    generated += itos[next_token]


print(generated)