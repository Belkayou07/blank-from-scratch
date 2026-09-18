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
    for index, character in enumerate(vocabulary)
}


itos = {
    index: character
    for character, index in stoi.items()
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

number_of_layers = 4

number_of_heads = 4

assert d_model % number_of_heads == 0

head_size = (
    d_model // number_of_heads
)

mlp_hidden = (
    4 * d_model
)

learning_rate = 3e-4

training_steps = 10000

evaluation_interval = 200

evaluation_batches = 30


print("\nMODEL CONFIGURATION")

print("Device:", device)
print("Context length:", context_length)
print("Embedding dimension:", d_model)
print("Attention heads:", number_of_heads)
print("Dimensions per head:", head_size)
print("Transformer layers:", number_of_layers)


# ============================================================
# 5. RANDOM MINI-BATCHES
# ============================================================

def get_batch(split_name):

    if split_name == "train":

        source = train_data

    else:

        source = validation_data


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

class CausalSelfAttention(nn.Module):

    def __init__(self):

        super().__init__()


        # ----------------------------------------
        # Q, K, V projections
        # ----------------------------------------

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


        # ----------------------------------------
        # Output projection
        # ----------------------------------------

        self.output = nn.Linear(
            d_model,
            d_model,
            bias=False
        )


        # ----------------------------------------
        # Causal mask
        # ----------------------------------------

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


        # ========================================
        # CREATE Q, K, V
        # ========================================

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)


        # ========================================
        # SPLIT INTO HEADS
        # ========================================

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


        # ========================================
        # ATTENTION SCORES
        # ========================================

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


        # ========================================
        # CAUSAL MASK
        # ========================================

        scores = scores.masked_fill(

            ~self.mask[
                :T,
                :T
            ],

            float("-inf")
        )


        # ========================================
        # SOFTMAX
        # ========================================

        attention = F.softmax(
            scores,
            dim=-1
        )


        # ========================================
        # APPLY ATTENTION TO VALUES
        # ========================================

        out = (
            attention
            @
            V
        )


        # ========================================
        # JOIN HEADS
        # ========================================

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

class FeedForward(nn.Module):

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

class TransformerBlock(nn.Module):

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


        # ----------------------------------------
        # Residual connection
        # ----------------------------------------

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

class MiniGPT(nn.Module):

    def __init__(self):

        super().__init__()


        # ========================================
        # TOKEN EMBEDDINGS
        # ========================================

        self.token_embedding = (
            nn.Embedding(
                vocab_size,
                d_model
            )
        )


        # ========================================
        # POSITION EMBEDDINGS
        # ========================================

        self.position_embedding = (
            nn.Embedding(
                context_length,
                d_model
            )
        )


        # ========================================
        # TRANSFORMER BLOCKS
        # ========================================

        self.blocks = nn.ModuleList([

            TransformerBlock()

            for _ in range(
                number_of_layers
            )
        ])


        # ========================================
        # FINAL NORMALIZATION
        # ========================================

        self.final_norm = (
            nn.LayerNorm(
                d_model
            )
        )


        # ========================================
        # VOCABULARY OUTPUT
        # ========================================

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


        # ========================================
        # TOKEN EMBEDDINGS
        # ========================================

        token_vectors = (
            self.token_embedding(X)
        )


        # ========================================
        # POSITION EMBEDDINGS
        # ========================================

        positions = torch.arange(
            T,
            device=X.device
        )


        position_vectors = (
            self.position_embedding(
                positions
            )
        )


        # ========================================
        # COMBINE TOKEN + POSITION
        # ========================================

        x = (
            token_vectors
            +
            position_vectors
        )


        # ========================================
        # PASS THROUGH ALL TRANSFORMER BLOCKS
        # ========================================

        attentions = []


        for block in self.blocks:

            if return_attention:

                x, attention = block(
                    x,
                    return_attention=True
                )

                attentions.append(
                    attention
                )

            else:

                x = block(x)


        # ========================================
        # FINAL NORMALIZATION
        # ========================================

        x = self.final_norm(x)


        # ========================================
        # VOCABULARY LOGITS
        # ========================================

        logits = self.output(x)


        # ========================================
        # LOSS
        # ========================================

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
                attentions
            )


        return (
            logits,
            loss
        )


# ============================================================
# 10. CREATE BLANK MODEL
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
    # Evaluation
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
    # Random training batch
    # ----------------------------------------

    X_batch, Y_batch = (
        get_batch("train")
    )


    # ----------------------------------------
    # Forward pass
    # ----------------------------------------

    logits, loss = model(
        X_batch,
        Y_batch
    )


    # ----------------------------------------
    # Clear previous gradients
    # ----------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )


    # ----------------------------------------
    # Backpropagation
    # ----------------------------------------

    loss.backward()


    # ----------------------------------------
    # AdamW parameter update
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
# 15. SAVE CHECKPOINT
# ============================================================
#
# IMPORTANT:
# Save BEFORE attention inspection and generation.
#
# That way, if something later crashes,
# the trained model still exists on disk.

CHECKPOINT_FILE = (
    "mini_gpt_stage2_checkpoint.pt"
)


torch.save(

    {
        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "training_steps":
            training_steps,

        "vocabulary":
            vocabulary,

        "stoi":
            stoi,

        "itos":
            itos,

        "config": {

            "context_length":
                context_length,

            "d_model":
                d_model,

            "number_of_layers":
                number_of_layers,

            "number_of_heads":
                number_of_heads,

            "vocab_size":
                vocab_size
        },

        "final_train_loss":
            losses["train"],

        "final_validation_loss":
            losses["validation"]
    },

    CHECKPOINT_FILE
)


print(
    "\nCheckpoint saved:",
    CHECKPOINT_FILE
)


# ============================================================
# 16. INSPECT ATTENTION
# ============================================================

inspection_text = "ROMEO:"


inspection_tokens = torch.tensor(

    [
        encode(
            inspection_text
        )
    ],

    dtype=torch.long,

    device=device
)


model.eval()


with torch.no_grad():

    _, _, attentions = model(

        inspection_tokens,

        return_attention=True
    )


print(
    "\nATTENTION FROM FINAL TOKEN "
    f"OF {inspection_text!r}:"
)


# We inspect:
#
# first Transformer layer
# and
# last Transformer layer.

for layer_index in [

    0,

    number_of_layers - 1
]:

    print(
        f"\nLAYER {layer_index + 1}"
    )


    # attentions is a LIST.
    #
    # attentions[layer_index]
    #
    # gives the tensor belonging
    # to that Transformer layer.

    final_attention = (

        attentions[
            layer_index
        ][
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
# 17. TEXT GENERATION
# ============================================================

@torch.no_grad()
def generate(
    prompt,
    number_of_characters=500
):

    model.eval()


    generated_tokens = torch.tensor(

        [
            encode(prompt)
        ],

        dtype=torch.long,

        device=device
    )


    for _ in range(
        number_of_characters
    ):

        # ----------------------------------------
        # Model may only see its context window
        # ----------------------------------------

        context = generated_tokens[
            :,
            -context_length:
        ]


        # ----------------------------------------
        # Forward pass
        # ----------------------------------------

        logits, _ = model(
            context
        )


        # ----------------------------------------
        # Prediction from final position only
        # ----------------------------------------

        logits = logits[
            :,
            -1,
            :
        ]


        # ----------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------

        probabilities = F.softmax(
            logits,
            dim=-1
        )


        # ----------------------------------------
        # Sample next token
        # ----------------------------------------

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )


        # ----------------------------------------
        # Append token
        # ----------------------------------------

        generated_tokens = torch.cat(

            [
                generated_tokens,
                next_token
            ],

            dim=1
        )


    return decode(
        generated_tokens[
            0
        ].tolist()
    )


# ============================================================
# 18. GENERATE SAMPLE
# ============================================================

print(
    "\nGENERATED TEXT:\n"
)


generated_text = generate(
    "ROMEO:",
    500
)


print(
    generated_text
)