import torch
import torch.nn.functional as F


# ==================================================
# DATA
# ==================================================

train_text = """
the cat sleeps on the mat.
the cat walks through the house.
the dog sleeps near the door.
the dog walks through the garden.
the cat sees the dog.
""".strip()


validation_text = """
the dog sees the cat.
""".strip()


# ==================================================
# VOCABULARY
# ==================================================

# Build the vocabulary from both pieces here because
# this experiment is about prediction/generalization,
# not handling unknown characters yet.

all_text = train_text + validation_text

vocabulary = sorted(set(all_text))
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


# ==================================================
# TURN TEXT INTO CONTEXT/TARGET EXAMPLES
# ==================================================

context_length = 3


def build_dataset(text):

    tokens = [
        stoi[c]
        for c in text
    ]

    X = []
    Y = []

    for i in range(len(tokens) - context_length):

        X.append(
            tokens[i:i + context_length]
        )

        Y.append(
            tokens[i + context_length]
        )

    return (
        torch.tensor(X, dtype=torch.long),
        torch.tensor(Y, dtype=torch.long)
    )


X_train, Y_train = build_dataset(train_text)

X_val, Y_val = build_dataset(validation_text)


print("Training examples:", len(X_train))
print("Validation examples:", len(X_val))


# ==================================================
# MODEL PARAMETERS
# ==================================================

torch.manual_seed(42)

embedding_dim = 4
hidden_size = 32


C = torch.randn(
    vocab_size,
    embedding_dim,
    requires_grad=True
)


W1 = (
    torch.randn(
        context_length * embedding_dim,
        hidden_size
    ) * 0.1
).requires_grad_()


b1 = torch.zeros(
    hidden_size,
    requires_grad=True
)


W2 = (
    torch.randn(
        hidden_size,
        vocab_size
    ) * 0.1
).requires_grad_()


b2 = torch.zeros(
    vocab_size,
    requires_grad=True
)


parameters = [
    C,
    W1,
    b1,
    W2,
    b2
]


print(
    "Trainable parameters:",
    sum(p.numel() for p in parameters)
)


# ==================================================
# FORWARD PASS
# ==================================================

def forward(X):

    embeddings = C[X]

    flattened = embeddings.reshape(
        len(X),
        context_length * embedding_dim
    )

    hidden = torch.tanh(
        flattened @ W1 + b1
    )

    logits = hidden @ W2 + b2

    return logits


# ==================================================
# TRAINING
# ==================================================

learning_rate = 0.1
epochs = 3000


for epoch in range(epochs):

    # -------------------------------
    # TRAINING FORWARD PASS
    # -------------------------------

    train_logits = forward(X_train)

    train_loss = F.cross_entropy(
        train_logits,
        Y_train
    )


    # -------------------------------
    # CLEAR GRADIENTS
    # -------------------------------

    for parameter in parameters:
        parameter.grad = None


    # -------------------------------
    # BACKPROPAGATION
    # -------------------------------

    train_loss.backward()


    # -------------------------------
    # UPDATE PARAMETERS
    # -------------------------------

    with torch.no_grad():

        for parameter in parameters:

            parameter -= (
                learning_rate
                * parameter.grad
            )


    # -------------------------------
    # VALIDATION
    # -------------------------------

    if epoch % 200 == 0:

        with torch.no_grad():

            validation_logits = forward(X_val)

            validation_loss = F.cross_entropy(
                validation_logits,
                Y_val
            )


        print(
            f"epoch={epoch:4d} | "
            f"train={train_loss.item():.4f} | "
            f"validation={validation_loss.item():.4f}"
        )


# ==================================================
# FINAL LOSSES
# ==================================================

with torch.no_grad():

    final_train_loss = F.cross_entropy(
        forward(X_train),
        Y_train
    )

    final_validation_loss = F.cross_entropy(
        forward(X_val),
        Y_val
    )


print()
print("FINAL")
print(
    "Training loss:  ",
    round(final_train_loss.item(), 4)
)

print(
    "Validation loss:",
    round(final_validation_loss.item(), 4)
)