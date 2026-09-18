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

number_of_heads = 4

assert d_model % number_of_heads == 0

head_size = (
    d_model // number_of_heads
)

mlp_hidden = 64


print("Attention heads:", number_of_heads)
print("Dimensions per head:", head_size)


# ==================================================
# PARAMETER CREATION
# ==================================================

def parameter(*shape, scale=0.1):

    return (
        torch.randn(*shape) * scale
    ).requires_grad_()


# ==================================================
# EMBEDDINGS
# ==================================================

token_embedding = parameter(
    vocab_size,
    d_model
)

position_embedding = parameter(
    context_length,
    d_model
)


# ==================================================
# MULTI-HEAD ATTENTION PARAMETERS
# ==================================================

W_Q = parameter(
    d_model,
    d_model
)

W_K = parameter(
    d_model,
    d_model
)

W_V = parameter(
    d_model,
    d_model
)

W_attention_out = parameter(
    d_model,
    d_model
)


# ==================================================
# LAYERNORM 1
# ==================================================

ln1_gain = torch.ones(
    d_model,
    requires_grad=True
)

ln1_bias = torch.zeros(
    d_model,
    requires_grad=True
)


# ==================================================
# MLP
# ==================================================

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


# ==================================================
# LAYERNORM 2
# ==================================================

ln2_gain = torch.ones(
    d_model,
    requires_grad=True
)

ln2_bias = torch.zeros(
    d_model,
    requires_grad=True
)


# ==================================================
# VOCABULARY OUTPUT
# ==================================================

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
# LAYERNORM
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
        /
        torch.sqrt(
            variance + 1e-5
        )
    )

    return (
        gain * normalized
        + bias
    )


# ==================================================
# FORWARD
# ==================================================

def forward(X, attention_mode="learned"):

    B, T = X.shape


    # ----------------------------------------------
    # TOKEN + POSITION EMBEDDINGS
    # ----------------------------------------------

    x = (
        token_embedding[X]
        +
        position_embedding[:T]
    )


    # ----------------------------------------------
    # CREATE Q, K, V
    # ----------------------------------------------

    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V


    # Initially:
    #
    # [B, T, 16]
    #
    # Split the 16 dimensions into:
    #
    # 4 heads × 4 dimensions.

    Q = Q.reshape(
        B,
        T,
        number_of_heads,
        head_size
    )

    K = K.reshape(
        B,
        T,
        number_of_heads,
        head_size
    )

    V = V.reshape(
        B,
        T,
        number_of_heads,
        head_size
    )


    # Change ordering to:
    #
    # [B, heads, T, head_size]

    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)


    # ----------------------------------------------
    # ATTENTION SCORES
    # ----------------------------------------------

    scores = (
        Q @ K.transpose(-2, -1)
    ) / (head_size ** 0.5)


    # Shape:
    #
    # [B, 4, T, T]
    #
    # Each head now owns a separate
    # T × T attention matrix.


    # ----------------------------------------------
    # CAUSAL MASK
    # ----------------------------------------------

    mask = causal_mask[:T, :T]

    scores = scores.masked_fill(
        mask,
        float("-inf")
    )


    # ----------------------------------------------
    # SOFTMAX
    # ----------------------------------------------

    if attention_mode == "learned":

        attention = F.softmax(
            scores,
            dim=-1
        )


    elif attention_mode == "uniform":

        # Every token attends equally to all positions
        # it is legally allowed to see.

        allowed = torch.tril(
            torch.ones(
                T,
                T,
                device=X.device
            )
        )

        uniform = (
            allowed
            /
            allowed.sum(
                dim=-1,
                keepdim=True
            )
        )

        # Shape [1, 1, T, T] lets PyTorch broadcast
        # this across every batch item and every head.

        attention = uniform[
            None,
            None,
            :,
            :
        ]


    elif attention_mode == "none":

        # Completely remove information coming
        # through the attention branch.

        attention = torch.zeros_like(
            scores
        )


    else:

        raise ValueError(
            f"Unknown attention mode: {attention_mode}"
        )


    # ----------------------------------------------
    # EACH HEAD MIXES ITS VALUES
    # ----------------------------------------------

    out = attention @ V


    # Current shape:
    #
    # [B, heads, T, head_size]
    #
    # Put sequence dimension back before heads.

    out = out.transpose(1, 2)


    # ----------------------------------------------
    # CONCATENATE HEADS
    # ----------------------------------------------

    out = out.reshape(
        B,
        T,
        d_model
    )


    # Mix information from all heads.

    out = (
        out @ W_attention_out
    )


    # ----------------------------------------------
    # RESIDUAL + NORMALIZATION
    # ----------------------------------------------

    x = x + out

    x = layer_norm(
        x,
        ln1_gain,
        ln1_bias
    )


    # ----------------------------------------------
    # MLP
    # ----------------------------------------------

    mlp = x @ W1 + b1

    mlp = F.relu(mlp)

    mlp = mlp @ W2 + b2


    x = x + mlp


    x = layer_norm(
        x,
        ln2_gain,
        ln2_bias
    )


    # ----------------------------------------------
    # VOCABULARY LOGITS
    # ----------------------------------------------

    logits = (
        x @ W_vocab
        + b_vocab
    )


    return logits, attention


# ==================================================
# TRAINING
# ==================================================

learning_rate = 0.05
epochs = 2500


for epoch in range(epochs):

    logits, attention = forward(X)


    loss = F.cross_entropy(
        logits.reshape(
            -1,
            vocab_size
        ),
        Y.reshape(-1)
    )


    for p in parameters:
        p.grad = None


    loss.backward()


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
# ATTENTION ABLATION TEST
# ==================================================

print("\nATTENTION ABLATION:")


with torch.no_grad():

    for mode in [
        "learned",
        "uniform",
        "none"
    ]:

        test_logits, _ = forward(
            X,
            attention_mode=mode
        )

        test_loss = F.cross_entropy(
            test_logits.reshape(
                -1,
                vocab_size
            ),
            Y.reshape(-1)
        )

        print(
            f"{mode:8s} attention | "
            f"loss = {test_loss.item():.6f}"
        )

# ==================================================
# INSPECT LEARNED ATTENTION
# ==================================================

inspection_text = "the cat "

inspection_tokens = torch.tensor(
    [[stoi[c] for c in inspection_text]],
    dtype=torch.long
)


with torch.no_grad():

    _, inspection_attention = forward(
        inspection_tokens
    )


print(
    "\nATTENTION FROM THE FINAL POSITION "
    "OF 'the cat ':"
)


# Batch 0
# Every head
# Final token position
#
# Shape becomes:
# [number_of_heads, T]

final_attention = (
    inspection_attention[
        0,
        :,
        -1,
        :
    ]
)


for head in range(number_of_heads):

    print(f"\nHead {head + 1}:")

    for position, character in enumerate(
        inspection_text
    ):

        percentage = (
            final_attention[
                head,
                position
            ].item()
            * 100
        )

        print(
            f"position {position} "
            f"{character!r}: "
            f"{percentage:6.2f}%"
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

        final_logits = logits[
            0,
            -1
        ]

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