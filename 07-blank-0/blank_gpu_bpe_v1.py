import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. FILES
# ============================================================

DATA_FILE = "tinyshakespeare.txt"

TOKENIZER_FILE = "bpe_tokenizer.json"


if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(DATA_FILE)


if not os.path.exists(TOKENIZER_FILE):
    raise FileNotFoundError(
        "bpe_tokenizer.json was not found. "
        "Run bpe_tokenizer.py first."
    )


# ============================================================
# 2. LOAD BPE TOKENIZER
# ============================================================

with open(
    TOKENIZER_FILE,
    "r",
    encoding="utf-8"
) as f:

    tokenizer_data = json.load(f)


BASE_VOCAB_SIZE = (
    tokenizer_data[
        "base_vocab_size"
    ]
)


merges = [
    tuple(merge)
    for merge
    in tokenizer_data["merges"]
]


vocab_size = (
    BASE_VOCAB_SIZE
    +
    len(merges)
)


# ------------------------------------------------------------
# Reconstruct token ID -> bytes
# ------------------------------------------------------------

vocab = {

    token_id: bytes(
        [token_id]
    )

    for token_id in range(
        BASE_VOCAB_SIZE
    )
}


for left, right, new_id in merges:

    vocab[new_id] = (
        vocab[left]
        +
        vocab[right]
    )


print(
    "BPE vocabulary size:",
    vocab_size
)


# ============================================================
# 3. TOKENIZER FUNCTIONS
# ============================================================

def merge_pair(
    token_ids,
    pair,
    new_token_id
):

    output = []

    i = 0


    while i < len(token_ids):

        if (
            i < len(token_ids) - 1
            and
            token_ids[i] == pair[0]
            and
            token_ids[i + 1] == pair[1]
        ):

            output.append(
                new_token_id
            )

            i += 2

        else:

            output.append(
                token_ids[i]
            )

            i += 1


    return output


def encode(text):

    token_ids = list(
        text.encode("utf-8")
    )


    for left, right, new_id in merges:

        token_ids = merge_pair(
            token_ids,
            (left, right),
            new_id
        )


    return token_ids


def decode(token_ids):

    byte_string = b"".join(

        vocab[token_id]

        for token_id
        in token_ids
    )


    return byte_string.decode(
        "utf-8",
        errors="replace"
    )


# ============================================================
# 4. LOAD TEXT
# ============================================================

with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    text = f.read()


print(
    "Characters in corpus:",
    len(text)
)


print(
    "Encoding complete corpus with our BPE tokenizer..."
)


token_ids = encode(text)


print(
    "BPE tokens in corpus:",
    len(token_ids)
)


print(
    "Average characters per token:",
    round(
        len(text)
        /
        len(token_ids),
        3
    )
)


data = torch.tensor(
    token_ids,
    dtype=torch.long
)


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

split = int(
    0.9 * len(data)
)


train_data = data[:split]

validation_data = data[split:]


print(
    "Training tokens:",
    len(train_data)
)

print(
    "Validation tokens:",
    len(validation_data)
)


# ============================================================
# 6. GPU
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "ROCm GPU is unavailable."
    )


device = torch.device(
    "cuda:0"
)


print(
    "\nGPU:",
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
        /
        1024**3,
        2
    ),
    "GB"
)


# ============================================================
# 7. CONFIGURATION
# ============================================================

torch.manual_seed(42)


batch_size = 32

context_length = 256

d_model = 256

number_of_heads = 8

number_of_layers = 6

mlp_hidden = (
    4 * d_model
)

head_size = (
    d_model
    // number_of_heads
)


assert (
    d_model
    % number_of_heads
    == 0
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
    "Context:",
    context_length,
    "BPE tokens"
)

print(
    "Embedding dimension:",
    d_model
)

print(
    "Layers:",
    number_of_layers
)

print(
    "Heads:",
    number_of_heads
)

print(
    "Head dimension:",
    head_size
)

print(
    "MLP hidden:",
    mlp_hidden
)


# ============================================================
# 8. TOKEN BYTE LENGTHS
# ============================================================
#
# Used to calculate bits-per-byte.
#
# Example:
#
# token "a"     -> 1 byte
# token "the "  -> 4 bytes
# token "ing"   -> 3 bytes

token_byte_lengths = torch.tensor(

    [
        len(vocab[token_id])

        for token_id in range(
            vocab_size
        )
    ],

    dtype=torch.float32,

    device=device
)


# ============================================================
# 9. MOVE DATA TO GPU
# ============================================================

train_data = train_data.to(
    device
)

validation_data = (
    validation_data.to(
        device
    )
)


# ============================================================
# 10. RANDOM BATCHES
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
        (batch_size,),
        device=device
    )


    offsets = torch.arange(
        context_length,
        device=device
    )


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
# 11. CAUSAL SELF-ATTENTION
# ============================================================

class CausalSelfAttention(
    nn.Module
):

    def __init__(self):

        super().__init__()


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


        self.output = nn.Linear(
            d_model,
            d_model,
            bias=False
        )


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
        x
    ):

        B, T, C = x.shape


        # ----------------------------------------------------
        # Q / K / V
        # ----------------------------------------------------

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)


        # ----------------------------------------------------
        # Split into heads
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        scores = (
            Q
            @
            K.transpose(-2, -1)
        )


        scores = (
            scores
            /
            math.sqrt(
                head_size
            )
        )


        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------

        scores = scores.masked_fill(

            ~self.mask[
                :T,
                :T
            ],

            float("-inf")
        )


        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        attention = F.softmax(
            scores,
            dim=-1
        )


        # ----------------------------------------------------
        # Weighted values
        # ----------------------------------------------------

        out = (
            attention
            @
            V
        )


        # ----------------------------------------------------
        # Reassemble heads
        # ----------------------------------------------------

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


        return self.output(
            out
        )


# ============================================================
# 12. FEED-FORWARD
# ============================================================

class FeedForward(
    nn.Module
):

    def __init__(self):

        super().__init__()


        self.network = (
            nn.Sequential(

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
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# 13. TRANSFORMER BLOCK
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
        x
    ):

        # Pre-norm attention

        x = (
            x
            +
            self.attention(
                self.norm1(x)
            )
        )


        # Pre-norm MLP

        x = (
            x
            +
            self.feed_forward(
                self.norm2(x)
            )
        )


        return x


# ============================================================
# 14. BLANK BPE TRANSFORMER
# ============================================================

class BlankBPETransformer(
    nn.Module
):

    def __init__(self):

        super().__init__()


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


        self.blocks = nn.ModuleList([

            TransformerBlock()

            for _ in range(
                number_of_layers
            )
        ])


        self.final_norm = (
            nn.LayerNorm(
                d_model
            )
        )


        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )


    def forward(
        self,
        X,
        targets=None
    ):

        B, T = X.shape


        if T > context_length:

            raise ValueError(
                "Input exceeds context window."
            )


        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Transformer stack
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(x)


        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        x = self.final_norm(x)


        # ----------------------------------------------------
        # Token predictions
        # ----------------------------------------------------

        logits = self.lm_head(x)


        loss = None


        if targets is not None:

            loss = F.cross_entropy(

                logits.reshape(
                    -1,
                    vocab_size
                ),

                targets.reshape(-1)
            )


        return (
            logits,
            loss
        )


# ============================================================
# 15. CREATE BLANK MODEL
# ============================================================

model = (
    BlankBPETransformer()
    .to(device)
)


parameter_count = sum(

    parameter.numel()

    for parameter
    in model.parameters()
)


print(
    "\nTRAINABLE PARAMETERS:",
    f"{parameter_count:,}"
)


# ============================================================
# 16. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=learning_rate,

    weight_decay=0.01
)


# ============================================================
# 17. EVALUATION
# ============================================================

@torch.no_grad()
def estimate_metrics():

    model.eval()


    results = {}


    for split_name in [
        "train",
        "validation"
    ]:

        total_nll = 0.0

        total_tokens = 0

        total_bytes = 0.0


        for _ in range(
            evaluation_batches
        ):

            X_batch, Y_batch = (
                get_batch(
                    split_name
                )
            )


            logits, _ = model(
                X_batch
            )


            flat_logits = (
                logits.reshape(
                    -1,
                    vocab_size
                )
            )


            flat_targets = (
                Y_batch.reshape(-1)
            )


            losses = F.cross_entropy(

                flat_logits,

                flat_targets,

                reduction="none"
            )


            total_nll += (
                losses.sum().item()
            )


            total_tokens += (
                flat_targets.numel()
            )


            total_bytes += (

                token_byte_lengths[
                    flat_targets
                ]

                .sum()

                .item()
            )


        average_loss = (
            total_nll
            /
            total_tokens
        )


        bits_per_byte = (

            total_nll
            /
            total_bytes
            /
            math.log(2)
        )


        results[
            split_name
        ] = {

            "loss":
                average_loss,

            "bits_per_byte":
                bits_per_byte
        }


    model.train()


    return results


# ============================================================
# 18. CHECKPOINTS
# ============================================================

BEST_FILE = (
    "blank_gpu_bpe_v1_best.pt"
)

LATEST_FILE = (
    "blank_gpu_bpe_v1_latest.pt"
)

FINAL_FILE = (
    "blank_gpu_bpe_v1_final.pt"
)


best_validation_bpb = (
    float("inf")
)


def save_checkpoint(
    filename,
    step,
    metrics
):

    torch.save(

        {
            "step":
                step,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "metrics":
                metrics,

            "tokenizer":
                tokenizer_data,

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
# 19. TRAIN
# ============================================================

print("\nTRAINING\n")


training_start = (
    time.perf_counter()
)


for step in range(
    training_steps
):


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    if (
        step
        % evaluation_interval
        == 0
    ):

        torch.cuda.synchronize()


        metrics = (
            estimate_metrics()
        )


        train_loss = (
            metrics[
                "train"
            ][
                "loss"
            ]
        )


        val_loss = (
            metrics[
                "validation"
            ][
                "loss"
            ]
        )


        train_bpb = (
            metrics[
                "train"
            ][
                "bits_per_byte"
            ]
        )


        val_bpb = (
            metrics[
                "validation"
            ][
                "bits_per_byte"
            ]
        )


        allocated = (

            torch.cuda.memory_allocated(
                0
            )

            /
            1024**3
        )


        reserved = (

            torch.cuda.memory_reserved(
                0
            )

            /
            1024**3
        )


        print(
            f"step={step:4d} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"train BPB={train_bpb:.4f} | "
            f"val BPB={val_bpb:.4f} | "
            f"VRAM={reserved:.2f} GB"
        )


        save_checkpoint(
            LATEST_FILE,
            step,
            metrics
        )


        # Use normalized validation BPB
        # to decide which checkpoint is best.

        if (
            val_bpb
            <
            best_validation_bpb
        ):

            best_validation_bpb = (
                val_bpb
            )


            save_checkpoint(
                BEST_FILE,
                step,
                metrics
            )


            print(
                "  -> new best checkpoint"
            )


    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    X_batch, Y_batch = (
        get_batch("train")
    )


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    _, loss = model(
        X_batch,
        Y_batch
    )


    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )


    loss.backward()


    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    optimizer.step()


# ============================================================
# 20. FINAL METRICS
# ============================================================

torch.cuda.synchronize()


training_end = (
    time.perf_counter()
)


metrics = estimate_metrics()


print("\nFINAL RESULTS")


print(
    "Training loss:",
    round(
        metrics[
            "train"
        ][
            "loss"
        ],
        4
    )
)


print(
    "Validation loss:",
    round(
        metrics[
            "validation"
        ][
            "loss"
        ],
        4
    )
)


print(
    "Training bits/byte:",
    round(
        metrics[
            "train"
        ][
            "bits_per_byte"
        ],
        4
    )
)


print(
    "Validation bits/byte:",
    round(
        metrics[
            "validation"
        ][
            "bits_per_byte"
        ],
        4
    )
)


print(
    "Best validation bits/byte:",
    round(
        best_validation_bpb,
        4
    )
)


print(
    "Training time:",
    round(
        training_end
        -
        training_start,
        2
    ),
    "seconds"
)


save_checkpoint(
    FINAL_FILE,
    training_steps,
    metrics
)


print(
    "\nFinal checkpoint saved:",
    FINAL_FILE
)


# ============================================================
# 21. LOAD BEST CHECKPOINT
# ============================================================

checkpoint = torch.load(
    BEST_FILE,
    map_location=device
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


print(
    "Loaded best checkpoint "
    f"from step {checkpoint['step']}"
)


# ============================================================
# 22. GENERATION
# ============================================================

@torch.no_grad()
def generate(
    prompt,
    number_of_tokens=500,
    temperature=0.8
):

    model.eval()


    prompt_tokens = encode(
        prompt
    )


    generated = torch.tensor(

        [
            prompt_tokens
        ],

        dtype=torch.long,

        device=device
    )


    for _ in range(
        number_of_tokens
    ):

        context = generated[
            :,
            -context_length:
        ]


        logits, _ = model(
            context
        )


        logits = (
            logits[
                :,
                -1,
                :
            ]
            /
            temperature
        )


        probabilities = F.softmax(
            logits,
            dim=-1
        )


        next_token = (
            torch.multinomial(
                probabilities,
                num_samples=1
            )
        )


        generated = torch.cat(
            [
                generated,
                next_token
            ],
            dim=1
        )


    return decode(
        generated[
            0
        ].tolist()
    )


# ============================================================
# 23. GENERATE TEXT
# ============================================================

print(
    "\nGENERATED TEXT:\n"
)


print(
    generate(
        "ROMEO:",
        number_of_tokens=500,
        temperature=0.8
    )
)