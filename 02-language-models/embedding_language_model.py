import torch
import torch.nn.functional as F


# ==================================================
# TRAINING TEXT
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
# VOCABULARY / TOKENIZER
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

print("Vocabulary size:", vocab_size)


# Convert complete text to token IDs.

tokens = torch.tensor(
    [stoi[c] for c in text],
    dtype=torch.long
)


# ==================================================
# BUILD TRAINING EXAMPLES
# ==================================================

context_length = 3

X = []
Y = []

for i in range(len(tokens) - context_length):

    context = tokens[i:i + context_length]

    target = tokens[i + context_length]

    X.append(context)
    Y.append(target)


X = torch.stack(X)

Y = torch.tensor(Y)


print("Training examples:", len(X))
print("X shape:", X.shape)
print("Y shape:", Y.shape)


# ==================================================
# MODEL PARAMETERS
# ==================================================

torch.manual_seed(42)

embedding_dim = 4
hidden_size = 32


# Embedding table:
#
# 20 characters × 4 numbers each

C = torch.randn(
    vocab_size,
    embedding_dim,
    requires_grad=True
)


# First neural-network layer.
#
# Input:
# 3 embeddings × 4 dimensions = 12 numbers
#
# Output:
# 32 hidden neurons

W1 = torch.randn(
    context_length * embedding_dim,
    hidden_size,
    requires_grad=True
) * 0.1

b1 = torch.zeros(
    hidden_size,
    requires_grad=True
)


# Output layer:
#
# 32 hidden values
#        ↓
# 20 possible next characters

W2 = torch.randn(
    hidden_size,
    vocab_size,
    requires_grad=True
) * 0.1

b2 = torch.zeros(
    vocab_size,
    requires_grad=True
)


# W1 and W2 are no longer leaf tensors because of * 0.1.
# Turn them into independent trainable tensors.

W1 = W1.detach().requires_grad_()
W2 = W2.detach().requires_grad_()


parameters = [C, W1, b1, W2, b2]

number_of_parameters = sum(
    parameter.numel()
    for parameter in parameters
)

print("Trainable parameters:", number_of_parameters)


# ==================================================
# TRAINING
# ==================================================

learning_rate = 0.1
epochs = 2000


for epoch in range(epochs):

    # ----------------------------------------------
    # EMBEDDING LOOKUP
    # ----------------------------------------------

    embeddings = C[X]

    # Shape:
    #
    # [number of examples, 3, 4]


    # ----------------------------------------------
    # COMBINE THE THREE EMBEDDINGS
    # ----------------------------------------------

    flattened = embeddings.reshape(
        len(X),
        context_length * embedding_dim
    )

    # Shape:
    #
    # [number of examples, 12]


    # ----------------------------------------------
    # HIDDEN LAYER
    # ----------------------------------------------

    hidden = torch.tanh(
        flattened @ W1 + b1
    )

    # Shape:
    #
    # [number of examples, 32]


    # ----------------------------------------------
    # OUTPUT LOGITS
    # ----------------------------------------------

    logits = hidden @ W2 + b2

    # Shape:
    #
    # [number of examples, 20]


    # ----------------------------------------------
    # CROSS-ENTROPY LOSS
    # ----------------------------------------------

    loss = F.cross_entropy(
        logits,
        Y
    )


    # ----------------------------------------------
    # BACKPROPAGATION
    # ----------------------------------------------

    for parameter in parameters:

        parameter.grad = None


    loss.backward()


    # ----------------------------------------------
    # GRADIENT DESCENT
    # ----------------------------------------------

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


context_string = "the"

context = [
    stoi[c]
    for c in context_string
]

generated = context_string


for _ in range(300):

    x = torch.tensor(
        [context],
        dtype=torch.long
    )


    # Embedding lookup.
    embedding = C[x]


    # Flatten the three embeddings.
    flattened = embedding.reshape(
        1,
        context_length * embedding_dim
    )


    # Hidden layer.
    hidden = torch.tanh(
        flattened @ W1 + b1
    )


    # Output scores.
    logits = hidden @ W2 + b2


    # Convert logits to probabilities.
    probabilities = F.softmax(
        logits,
        dim=1
    )


    # Randomly sample according to those probabilities.
    next_token = torch.multinomial(
        probabilities,
        num_samples=1
    ).item()


    next_character = itos[next_token]

    generated += next_character


    # Slide the context window.
    context = context[1:] + [next_token]


print(generated)