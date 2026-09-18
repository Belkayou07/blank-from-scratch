import os
import urllib.request
import time

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

    print("Downloading Tiny Shakespeare...")

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


def decode(token_ids):

    return "".join(
        itos[token]
        for token in token_ids
    )


print("Vocabulary size:", vocab_size)


# ============================================================
# 3. DEVICE
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "ROCm GPU is not available."
    )


device = torch.device("cuda:0")


print("\nGPU INFORMATION")

print(
    "Device:",
    torch.cuda.get_device_name(0)
)

print(
    "HIP:",
    torch.version.hip
)

print(
    "VRAM:",
    round(
        torch.cuda.get_device_properties(
            0
        ).total_memory
        / 1024**3,
        2
    ),
    "GB"
)


# ============================================================
# 4. DATASET TO TOKEN IDs
# ============================================================

data = torch.tensor(
    encode(text),
    dtype=torch.long
)


split = int(
    0.9 * len(data)
)


train_data = data[:split]

validation_data = data[split:]


print(
    "\nTraining characters:",
    len(train_data)
)

print(
    "Validation characters:",
    len(validation_data)
)


# ============================================================
# 5. MODEL CONFIGURATION
# ============================================================

torch.manual_seed(42)


batch_size = 32

context_length = 256

d_model = 256

number_of_heads = 8

number_of_layers = 6

mlp_hidden = 4 * d_model


assert (
    d_model
    % number_of_heads
    == 0
)


head_size = (
    d_model
    // number_of_heads
)


learning_rate = 3e-4

training_steps = 3000

evaluation_interval = 200

evaluation_batches = 20


print("\nMODEL CONFIGURATION")

print(
    "Batch size:",
    batch_size
)

print(
    "Context length:",
    context_length
)

print(
    "Embedding dimension:",
    d_model
)

print(
    "Transformer layers:",
    number_of_layers
)

print(
    "Attention heads:",
    number_of_heads
)

print(
    "Dimensions per head:",
    head_size
)

print(
    "MLP hidden size:",
    mlp_hidden
)


# ============================================================
# 6. MOVE DATASET TO GPU
# ============================================================
#
# Tiny Shakespeare is very small compared with our 16 GB VRAM.
#
# Keeping it on the GPU avoids repeatedly copying mini-batches
# from CPU RAM.

train_data = train_data.to(device)

validation_data = (
    validation_data.to(device)
)


# ============================================================
# 7. RANDOM MINI-BATCHES
# ============================================================

def get_batch(split_name):

    if split_name == "train":

        source = train_data

    else:

        source = validation_data


    # Random starting positions.

    starts = torch.randint(
        0,
        len(source)
        - context_length
        - 1,
        (batch_size,),
        device=device
    )


    # [0, 1, 2, ..., 255]

    offsets = torch.arange(
        context_length,
        device=device
    )


    # Creates:
    #
    # [batch_size, context_length]
    #
    # without manually looping through
    # every training sequence.

    positions = (
        starts[:, None]
        +
        offsets[None, :]
    )


    X = source[
        positions
    ]


    Y = source[
        positions + 1
    ]


    return X, Y


# ============================================================
# 8. MULTI-HEAD CAUSAL SELF-ATTENTION
# ============================================================

class CausalSelfAttention(
    nn.Module
):

    def __init__(self):

        super().__init__()


        # ----------------------------------------
        # Q, K, V
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
        # Q / K / V PROJECTIONS
        # ========================================

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)


        # ========================================
        # SPLIT INTO HEADS
        # ========================================

        Q = (
            Q
            .reshape(
                B,
                T,
                number_of_heads,
                head_size
            )
            .transpose(1, 2)
        )


        K = (
            K
            .reshape(
                B,
                T,
                number_of_heads,
                head_size
            )
            .transpose(1, 2)
        )


        V = (
            V
            .reshape(
                B,
                T,
                number_of_heads,
                head_size
            )
            .transpose(1, 2)
        )


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
        # ATTENTION PROBABILITIES
        # ========================================

        attention = F.softmax(
            scores,
            dim=-1
        )


        # ========================================
        # WEIGHTED VALUES
        # ========================================

        out = (
            attention
            @
            V
        )


        # ========================================
        # REASSEMBLE HEADS
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
# 9. FEED-FORWARD NETWORK
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
# 10. TRANSFORMER BLOCK
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
# 11. BLANK GPU TRANSFORMER
# ============================================================

class BlankTransformer(
    nn.Module
):

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
        # LANGUAGE-MODEL HEAD
        # ========================================

        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )


    def forward(
        self,
        X,
        targets=None,
        return_attention=False
    ):

        B, T = X.shape


        if T > context_length:

            raise ValueError(
                "Sequence is longer than "
                "the model context window."
            )


        # ========================================
        # EMBEDDINGS
        # ========================================

        token_vectors = (
            self.token_embedding(X)
        )


        positions = torch.arange(
            T,
            device=X.device
        )


        position_vectors = (
            self.position_embedding(
                positions
            )
        )


        x = (
            token_vectors
            +
            position_vectors
        )


        # ========================================
        # TRANSFORMER STACK
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
        # LOGITS
        # ========================================

        logits = self.lm_head(x)


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
# 12. CREATE COMPLETELY BLANK MODEL
# ============================================================

model = BlankTransformer().to(
    device
)


number_of_parameters = sum(

    parameter.numel()

    for parameter
    in model.parameters()
)


print(
    "\nTRAINABLE PARAMETERS:",
    f"{number_of_parameters:,}"
)


# ============================================================
# 13. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=learning_rate,

    weight_decay=0.01
)


# ============================================================
# 14. EVALUATION
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

            X_batch, Y_batch = (
                get_batch(
                    split_name
                )
            )


            _, loss = model(
                X_batch,
                Y_batch
            )


            losses.append(
                loss.item()
            )


        results[
            split_name
        ] = (

            sum(losses)
            /
            len(losses)
        )


    model.train()


    return results


# ============================================================
# 15. CHECKPOINT SAVING
# ============================================================

CHECKPOINT_FILE = (
    "blank_gpu_v1_best.pt"
)


LATEST_FILE = (
    "blank_gpu_v1_latest.pt"
)


best_validation_loss = float("inf")


def save_checkpoint(
    filename,
    step,
    train_loss,
    validation_loss
):

    torch.save(

        {
            "step":
                step,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_loss":
                train_loss,

            "validation_loss":
                validation_loss,

            "vocabulary":
                vocabulary,

            "stoi":
                stoi,

            "itos":
                itos,

            "config": {

                "vocab_size":
                    vocab_size,

                "context_length":
                    context_length,

                "d_model":
                    d_model,

                "number_of_heads":
                    number_of_heads,

                "number_of_layers":
                    number_of_layers,

                "mlp_hidden":
                    mlp_hidden
            }
        },

        filename
    )


# ============================================================
# 16. TRAINING
# ============================================================

print("\nTRAINING\n")


model.train()


training_start = (
    time.perf_counter()
)


tokens_since_report = 0


for step in range(
    training_steps
):


    # ========================================
    # EVALUATION
    # ========================================

    if (
        step
        % evaluation_interval
        == 0
    ):

        torch.cuda.synchronize()


        losses = estimate_loss()


        allocated = (
            torch.cuda.memory_allocated(0)
            /
            1024**3
        )


        reserved = (
            torch.cuda.memory_reserved(0)
            /
            1024**3
        )


        print(
            f"step={step:4d} | "
            f"train={losses['train']:.4f} | "
            f"validation={losses['validation']:.4f} | "
            f"allocated={allocated:.2f} GB | "
            f"reserved={reserved:.2f} GB"
        )


        # ----------------------------------------
        # Save latest checkpoint.
        # ----------------------------------------

        save_checkpoint(

            LATEST_FILE,

            step,

            losses["train"],

            losses["validation"]
        )


        # ----------------------------------------
        # Save best validation checkpoint.
        # ----------------------------------------

        if (
            losses["validation"]
            <
            best_validation_loss
        ):

            best_validation_loss = (
                losses["validation"]
            )


            save_checkpoint(

                CHECKPOINT_FILE,

                step,

                losses["train"],

                losses["validation"]
            )


            print(
                "  -> new best checkpoint saved"
            )


    # ========================================
    # GET RANDOM MINI-BATCH
    # ========================================

    X_batch, Y_batch = (
        get_batch("train")
    )


    # ========================================
    # FORWARD PASS
    # ========================================

    _, loss = model(
        X_batch,
        Y_batch
    )


    # ========================================
    # CLEAR PREVIOUS GRADIENTS
    # ========================================

    optimizer.zero_grad(
        set_to_none=True
    )


    # ========================================
    # BACKPROPAGATION
    # ========================================

    loss.backward()


    # ========================================
    # UPDATE PARAMETERS
    # ========================================

    optimizer.step()


    tokens_since_report += (
        batch_size
        *
        context_length
    )


# ============================================================
# 17. FINAL EVALUATION
# ============================================================

torch.cuda.synchronize()


training_end = (
    time.perf_counter()
)


losses = estimate_loss()


print("\nFINAL RESULTS")

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


print(
    "Best validation loss:",
    round(
        best_validation_loss,
        4
    )
)


print(
    "GPU allocated:",
    round(
        torch.cuda.memory_allocated(0)
        / 1024**3,
        2
    ),
    "GB"
)


print(
    "GPU reserved:",
    round(
        torch.cuda.memory_reserved(0)
        / 1024**3,
        2
    ),
    "GB"
)


# Save final state too.

save_checkpoint(

    "blank_gpu_v1_final.pt",

    training_steps,

    losses["train"],

    losses["validation"]
)


print(
    "\nFinal checkpoint saved:",
    "blank_gpu_v1_final.pt"
)


# ============================================================
# 18. LOAD BEST MODEL
# ============================================================
#
# Generation should use our best validation checkpoint,
# not blindly use the final training step.

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=device
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


print(
    "\nLoaded best checkpoint "
    f"from step {checkpoint['step']}"
)


# ============================================================
# 19. GENERATION
# ============================================================

@torch.no_grad()
def generate(
    prompt,
    number_of_characters=1000,
    temperature=0.8
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
        # Context window
        # ----------------------------------------

        context = (
            generated_tokens[
                :,
                -context_length:
            ]
        )


        # ----------------------------------------
        # Forward
        # ----------------------------------------

        logits, _ = model(
            context
        )


        # ----------------------------------------
        # Final position
        # ----------------------------------------

        logits = (
            logits[
                :,
                -1,
                :
            ]
        )


        # ----------------------------------------
        # Temperature
        # ----------------------------------------

        logits = (
            logits
            /
            temperature
        )


        # ----------------------------------------
        # Probabilities
        # ----------------------------------------

        probabilities = F.softmax(
            logits,
            dim=-1
        )


        # ----------------------------------------
        # Sample
        # ----------------------------------------

        next_token = (
            torch.multinomial(
                probabilities,
                num_samples=1
            )
        )


        generated_tokens = (
            torch.cat(
                [
                    generated_tokens,
                    next_token
                ],
                dim=1
            )
        )


    return decode(
        generated_tokens[
            0
        ].tolist()
    )


# ============================================================
# 20. GENERATE SAMPLE
# ============================================================

print(
    "\nGENERATED TEXT:\n"
)


print(
    generate(
        "ROMEO:",
        number_of_characters=1000,
        temperature=0.8
    )
)