import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. DATA
# ============================================================

DATA_FILE = "tinyshakespeare.txt"

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


if not os.path.exists(DATA_FILE):

    print("Downloading training corpus...")

    urllib.request.urlretrieve(
        DATA_URL,
        DATA_FILE
    )


with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    text = f.read()


print("Characters in corpus:", len(text))


# ============================================================
# 2. CHARACTER TOKENIZER
# ============================================================

vocabulary = sorted(set(text))

vocab_size = len(vocabulary)


stoi = {
    character: index
    for index, character
    in enumerate(vocabulary)
}


itos = {
    index: character
    for character, index
    in stoi.items()
}


def encode(string):

    return [
        stoi[c]
        for c in string
    ]


def decode(tokens):

    return "".join(
        itos[token]
        for token in tokens
    )


print("Vocabulary size:", vocab_size)


# Entire corpus becomes token IDs.

data = torch.tensor(
    encode(text),
    dtype=torch.long
)


# ============================================================
# 3. TRAIN / VALIDATION SPLIT
# ============================================================

split = int(
    0.9 * len(data)
)


train_data = data[:split]

validation_data = data[split:]


print(
    "Training characters:",
    len(train_data)
)

print(
    "Validation characters:",
    len(validation_data)
)


# ============================================================
# 4. TRAINING CONFIGURATION
# ============================================================

torch.manual_seed(42)


device = "cpu"

batch_size = 32

context_length = 64

d_model = 64

number_of_heads = 4

head_size = (
    d_model // number_of_heads
)

mlp_hidden = (
    4 * d_model
)

learning_rate = 3e-4

training_steps = 2000

evaluation_interval = 200

evaluation_batches = 30


print("\nMODEL CONFIGURATION")

print("Device:", device)
print("Context length:", context_length)
print("Embedding dimension:", d_model)
print("Attention heads:", number_of_heads)
print("Dimensions per head:", head_size)


# ============================================================
# 5. RANDOM MINI-BATCHES
# ============================================================

def get_batch(split_name):

    source = (
        train_data
        if split_name == "train"
        else validation_data
    )


    # Pick random starting locations.

    starts = torch.randint(
        0,
        len(source)
        - context_length
        - 1,
        (batch_size,)
    )


    X = torch.stack([
        source[
            i:
            i + context_length
        ]

        for i in starts
    ])


    Y = torch.stack([
        source[
            i + 1:
            i + context_length + 1
        ]

        for i in starts
    ])


    return (
        X.to(device),
        Y.to(device)
    )


# ============================================================
# 6. CAUSAL MULTI-HEAD SELF-ATTENTION
# ============================================================

class CausalSelfAttention(
    nn.Module
):

    def __init__(self):

        super().__init__()


        # Same Q / K / V matrices that
        # we previously built manually.

        self.query = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.key = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.value = nn.Linear(
            d_model,
            d_model,
            bias=False
        )


        # Mix the concatenated heads.

        self.output = nn.Linear(
            d_model,
            d_model,
            bias=False
        )


        # Causal mask.

        mask = torch.tril(
            torch.ones(
                context_length,
                context_length,
                dtype=torch.bool
            )
        )


        self.register_buffer(
            "mask",
            mask
        )


    def forward(
        self,
        x,
        return_attention=False
    ):

        B, T, C = x.shape


        # ----------------------------------------
        # Q, K, V
        # ----------------------------------------

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)


        # ----------------------------------------
        # SPLIT INTO HEADS
        # ----------------------------------------

        Q = Q.reshape(
            B,
            T,
            number_of_heads,
            head_size
        ).transpose(1, 2)


        K = K.reshape(
            B,
            T,
            number_of_heads,
            head_size
        ).transpose(1, 2)


        V = V.reshape(
            B,
            T,
            number_of_heads,
            head_size
        ).transpose(1, 2)


        # Shapes:
        #
        # [B, heads, T, head_size]


        # ----------------------------------------
        # QK ATTENTION SCORES
        # ----------------------------------------

        scores = (
            Q
            @
            K.transpose(-2, -1)
        )


        scores = (
            scores
            /
            (head_size ** 0.5)
        )


        # ----------------------------------------
        # CAUSAL MASK
        # ----------------------------------------

        scores = scores.masked_fill(

            ~self.mask[
                :T,
                :T
            ],

            float("-inf")
        )


        # ----------------------------------------
        # SOFTMAX
        # ----------------------------------------

        attention = F.softmax(
            scores,
            dim=-1
        )


        # ----------------------------------------
        # COLLECT VALUES
        # ----------------------------------------

        out = (
            attention
            @
            V
        )


        # ----------------------------------------
        # JOIN HEADS
        # ----------------------------------------

        out = (
            out
            .transpose(1, 2)
            .contiguous()
            .reshape(
                B,
                T,
                d_model
            )
        )


        out = self.output(out)


        if return_attention:

            return (
                out,
                attention
            )


        return out


# ============================================================
# 7. FEED-FORWARD NETWORK
# ============================================================

class FeedForward(
    nn.Module
):

    def __init__(self):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                d_model,
                mlp_hidden
            ),

            nn.GELU(),

            nn.Linear(
                mlp_hidden,
                d_model
            )
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# 8. TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(
    nn.Module
):

    def __init__(self):

        super().__init__()


        self.norm1 = nn.LayerNorm(
            d_model
        )


        self.attention = (
            CausalSelfAttention()
        )


        self.norm2 = nn.LayerNorm(
            d_model
        )


        self.feed_forward = (
            FeedForward()
        )


    def forward(
        self,
        x,
        return_attention=False
    ):

        # ========================================
        # PRE-NORM ATTENTION
        # ========================================

        normalized = self.norm1(x)


        if return_attention:

            attention_output, weights = (
                self.attention(
                    normalized,
                    return_attention=True
                )
            )

        else:

            attention_output = (
                self.attention(
                    normalized
                )
            )


        # Residual connection.

        x = (
            x
            +
            attention_output
        )


        # ========================================
        # PRE-NORM MLP
        # ========================================

        x = (
            x
            +
            self.feed_forward(
                self.norm2(x)
            )
        )


        if return_attention:

            return (
                x,
                weights
            )


        return x


# ============================================================
# 9. MINI GPT
# ============================================================

class MiniGPT(
    nn.Module
):

    def __init__(self):

        super().__init__()


        # ----------------------------------------
        # EMBEDDINGS
        # ----------------------------------------

        self.token_embedding = (
            nn.Embedding(
                vocab_size,
                d_model
            )
        )


        self.position_embedding = (
            nn.Embedding(
                context_length,
                d_model
            )
        )


        # For now:
        #
        # ONE Transformer block.
        #
        # We'll stack them later.

        self.block = (
            TransformerBlock()
        )


        self.final_norm = (
            nn.LayerNorm(
                d_model
            )
        )


        self.output = (
            nn.Linear(
                d_model,
                vocab_size
            )
        )


    def forward(
        self,
        X,
        targets=None,
        return_attention=False
    ):

        B, T = X.shape


        # ----------------------------------------
        # TOKEN EMBEDDINGS
        # ----------------------------------------

        token_vectors = (
            self.token_embedding(X)
        )


        # ----------------------------------------
        # POSITION EMBEDDINGS
        # ----------------------------------------

        positions = torch.arange(
            T,
            device=X.device
        )


        position_vectors = (
            self.position_embedding(
                positions
            )
        )


        # ----------------------------------------
        # COMBINE
        # ----------------------------------------

        x = (
            token_vectors
            +
            position_vectors
        )


        # ----------------------------------------
        # TRANSFORMER BLOCK
        # ----------------------------------------

        if return_attention:

            x, attention = (
                self.block(
                    x,
                    return_attention=True
                )
            )

        else:

            x = self.block(x)


        # ----------------------------------------
        # FINAL NORMALIZATION
        # ----------------------------------------

        x = self.final_norm(x)


        # ----------------------------------------
        # VOCABULARY LOGITS
        # ----------------------------------------

        logits = self.output(x)


        # ----------------------------------------
        # LOSS
        # ----------------------------------------

        loss = None


        if targets is not None:

            loss = F.cross_entropy(

                logits.reshape(
                    -1,
                    vocab_size
                ),

                targets.reshape(-1)
            )


        if return_attention:

            return (
                logits,
                loss,
                attention
            )


        return (
            logits,
            loss
        )


# ============================================================
# 10. CREATE A BLANK MODEL
# ============================================================

model = MiniGPT().to(device)


number_of_parameters = sum(

    parameter.numel()

    for parameter
    in model.parameters()
)


print(
    "\nTrainable parameters:",
    f"{number_of_parameters:,}"
)


# ============================================================
# 11. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# 12. EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss():

    model.eval()


    results = {}


    for split_name in [
        "train",
        "validation"
    ]:

        losses = []


        for _ in range(
            evaluation_batches
        ):

            xb, yb = get_batch(
                split_name
            )


            _, loss = model(
                xb,
                yb
            )


            losses.append(
                loss.item()
            )


        results[split_name] = (
            sum(losses)
            /
            len(losses)
        )


    model.train()


    return results


# ============================================================
# 13. TRAINING
# ============================================================

print("\nTRAINING\n")


for step in range(
    training_steps
):


    # ----------------------------------------
    # PERIODIC EVALUATION
    # ----------------------------------------

    if (
        step
        % evaluation_interval
        == 0
    ):

        losses = estimate_loss()


        print(
            f"step={step:4d} | "
            f"train={losses['train']:.4f} | "
            f"validation={losses['validation']:.4f}"
        )


    # ----------------------------------------
    # RANDOM BATCH
    # ----------------------------------------

    X_batch, Y_batch = (
        get_batch("train")
    )


    # ----------------------------------------
    # FORWARD PASS
    # ----------------------------------------

    logits, loss = model(
        X_batch,
        Y_batch
    )


    # ----------------------------------------
    # CLEAR GRADIENTS
    # ----------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )


    # ----------------------------------------
    # BACKPROPAGATION
    # ----------------------------------------

    loss.backward()


    # ----------------------------------------
    # PARAMETER UPDATE
    # ----------------------------------------

    optimizer.step()


# ============================================================
# 14. FINAL EVALUATION
# ============================================================

losses = estimate_loss()


print("\nFINAL")

print(
    "Training loss:",
    round(
        losses["train"],
        4
    )
)

print(
    "Validation loss:",
    round(
        losses["validation"],
        4
    )
)


# ============================================================
# 15. INSPECT ATTENTION
# ============================================================

inspection_text = (
    "ROMEO:"
)


inspection_tokens = torch.tensor(
    [encode(inspection_text)],
    dtype=torch.long,
    device=device
)


model.eval()


with torch.no_grad():

    _, _, attention = model(
        inspection_tokens,
        return_attention=True
    )


print(
    "\nATTENTION FROM FINAL TOKEN "
    f"OF {inspection_text!r}:"
)


final_attention = (
    attention[
        0,
        :,
        -1,
        :
    ]
)


for head in range(
    number_of_heads
):

    print(
        f"\nHead {head + 1}:"
    )


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
            f"{position:2d} "
            f"{character!r}: "
            f"{percentage:6.2f}%"
        )


# ============================================================
# 16. TEXT GENERATION
# ============================================================

@torch.no_grad()
def generate(
    prompt,
    number_of_characters=500
):

    model.eval()


    tokens = torch.tensor(
        [encode(prompt)],
        dtype=torch.long,
        device=device
    )


    for _ in range(
        number_of_characters
    ):


        # Model can see only its context window.

        context = tokens[
            :,
            -context_length:
        ]


        logits, _ = model(
            context
        )


        # Only final position predicts
        # the next character.

        logits = logits[
            :,
            -1,
            :
        ]


        probabilities = F.softmax(
            logits,
            dim=-1
        )


        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )


        tokens = torch.cat(
            [
                tokens,
                next_token
            ],
            dim=1
        )


    return decode(
        tokens[0].tolist()
    )


print("\nGENERATED TEXT:\n")

print(
    generate(
        "ROMEO:",
        500
    )
)