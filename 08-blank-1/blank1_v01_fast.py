import json
import math
import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer


# ============================================================
# 1. FILES
# ============================================================

DATA_FILE = "blank_dataset_v2_tokens.bin"

METADATA_FILE = "blank_dataset_v2_tokens_meta.json"

TOKENIZER_FILE = "blank_tokenizer_v1.json"


BEST_CHECKPOINT = "blank1_v01_best.pt"

LATEST_CHECKPOINT = "blank1_v01_latest.pt"

FINAL_CHECKPOINT = "blank1_v01_final.pt"


# ============================================================
# 2. REQUIRED FILES
# ============================================================

for filename in [
    DATA_FILE,
    METADATA_FILE,
    TOKENIZER_FILE,
]:

    if not os.path.exists(filename):

        raise FileNotFoundError(
            filename
        )


# ============================================================
# 3. METADATA
# ============================================================

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8",
) as f:

    metadata = json.load(f)


vocab_size = metadata[
    "vocab_size"
]

total_tokens = metadata[
    "total_tokens"
]

train_tokens = metadata[
    "train_tokens"
]

validation_tokens = metadata[
    "validation_tokens"
]


print(
    "Total tokens:",
    f"{total_tokens:,}"
)

print(
    "Training tokens:",
    f"{train_tokens:,}"
)

print(
    "Validation tokens:",
    f"{validation_tokens:,}"
)

print(
    "Vocabulary:",
    f"{vocab_size:,}"
)


# ============================================================
# 4. TOKENIZER
# ============================================================

tokenizer = Tokenizer.from_file(
    TOKENIZER_FILE
)


# ============================================================
# 5. GPU
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "ROCm GPU unavailable."
    )


device = torch.device(
    "cuda:0"
)


print(
    "\nGPU:",
    torch.cuda.get_device_name(0)
)

print(
    "PyTorch:",
    torch.__version__
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
        2,
    ),
    "GB",
)


# ============================================================
# 6. RANDOM SEED
# ============================================================

SEED = 42


torch.manual_seed(
    SEED
)

torch.cuda.manual_seed_all(
    SEED
)


# ============================================================
# 7. BLANK-1 ARCHITECTURE
# ============================================================

context_length = 512

batch_size = 16


d_model = 512

number_of_layers = 10

number_of_heads = 8

head_size = 64


mlp_hidden = 1365


assert (
    d_model
    ==
    number_of_heads
    *
    head_size
)


assert (
    head_size
    %
    2
    ==
    0
)


tokens_per_update = (

    batch_size

    *

    context_length
)


# ============================================================
# 8. TRAINING CONFIG
# ============================================================

TARGET_TOKEN_PREDICTIONS = (
    285_736_960
)


training_updates = math.ceil(

    TARGET_TOKEN_PREDICTIONS

    /

    tokens_per_update
)


actual_token_predictions = (

    training_updates

    *

    tokens_per_update
)


maximum_learning_rate = (
    2.5e-4
)

minimum_learning_rate = (
    2.5e-5
)


warmup_updates = 1300


weight_decay = 0.1

gradient_clip = 1.0


# ------------------------------------------------------------
# Evaluation is less frequent than previous experiments
# to avoid unnecessary wall-clock overhead.
# ------------------------------------------------------------

evaluation_interval = 2000

evaluation_batches = 20


# Final evaluation is more thorough.

final_evaluation_batches = 50


RESUME = True


BLANK0_BEST_VALIDATION = (
    3.0686864535013836
)


print(
    "\nBLANK-1 v0.1 — FULL TRAINING"
)

print(
    "Context:",
    context_length
)

print(
    "Batch:",
    batch_size
)

print(
    "Tokens/update:",
    f"{tokens_per_update:,}"
)

print(
    "Width:",
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
    "SwiGLU hidden:",
    mlp_hidden
)

print(
    "Training updates:",
    f"{training_updates:,}"
)

print(
    "Target token predictions:",
    f"{TARGET_TOKEN_PREDICTIONS:,}"
)

print(
    "Actual token predictions:",
    f"{actual_token_predictions:,}"
)

print(
    "Dataset equivalents:",
    round(
        actual_token_predictions
        /
        train_tokens,
        4,
    )
)

print(
    "Precision: BF16 autocast"
)

print(
    "Attention: SDPA"
)

print(
    "Optimizer: fused AdamW"
)

print(
    "Execution: eager"
)


# ============================================================
# 9. LOAD TOKEN DATASET
# ============================================================
#
# Keep corpus INT32 on GPU:
#
# ~1.09 GB
#
# rather than expanding the entire corpus to int64:
#
# ~2.17 GB
#
# Only the selected mini-batch becomes int64/long.
# ============================================================

print(
    "\nLoading binary token dataset..."
)


binary_tokens = np.memmap(

    DATA_FILE,

    dtype=np.uint16,

    mode="r",
)


if len(binary_tokens) != total_tokens:

    raise RuntimeError(
        "Dataset token count mismatch."
    )


print(
    "Converting corpus to INT32..."
)


cpu_array = np.asarray(

    binary_tokens,

    dtype=np.int32,
)


cpu_tokens = torch.from_numpy(
    cpu_array
)


print(
    "Moving corpus to GPU..."
)


all_tokens = cpu_tokens.to(
    device
)


del cpu_tokens
del cpu_array
del binary_tokens


train_data = all_tokens[
    :train_tokens
]


validation_data = all_tokens[
    train_tokens:
]


dataset_memory_gb = (

    all_tokens.numel()

    *

    all_tokens.element_size()

    /

    1024**3
)


print(
    "Dataset GPU memory:",
    round(
        dataset_memory_gb,
        2,
    ),
    "GB",
)


# ============================================================
# 10. BATCH SAMPLING
# ============================================================

batch_offsets = torch.arange(

    context_length,

    device=device,
)


def get_batch(
    split,
):

    if split == "train":

        source = train_data

    elif split == "validation":

        source = validation_data

    else:

        raise ValueError(
            split
        )


    starts = torch.randint(

        low=0,

        high=(
            len(source)
            -
            context_length
            -
            1
        ),

        size=(
            batch_size,
        ),

        device=device,
    )


    positions = (

        starts[:, None]

        +

        batch_offsets[
            None,
            :
        ]
    )


    # Dataset itself remains INT32.
    #
    # Embedding indices and CE targets become
    # INT64 only for the current mini-batch.

    X = source[
        positions
    ].long()


    Y = source[
        positions + 1
    ].long()


    return X, Y


# ============================================================
# 11. RMSNORM
# ============================================================

class RMSNorm(
    nn.Module
):

    def __init__(
        self,
        dimension,
        epsilon=1e-5,
    ):

        super().__init__()


        self.weight = nn.Parameter(

            torch.ones(
                dimension
            )
        )


        self.epsilon = epsilon


    def forward(
        self,
        x,
    ):

        mean_square = torch.mean(

            x * x,

            dim=-1,

            keepdim=True,
        )


        normalized = (

            x

            *

            torch.rsqrt(

                mean_square

                +

                self.epsilon
            )
        )


        return (

            normalized

            *

            self.weight
        )


# ============================================================
# 12. ROTARY POSITION EMBEDDING
# ============================================================

def apply_rotary(
    x,
    cosine,
    sine,
):

    even = x[
        ...,
        0::2
    ]


    odd = x[
        ...,
        1::2
    ]


    new_even = (

        even * cosine

        -

        odd * sine
    )


    new_odd = (

        even * sine

        +

        odd * cosine
    )


    return torch.stack(

        [
            new_even,
            new_odd,
        ],

        dim=-1,

    ).flatten(
        -2
    )


class RotaryEmbedding(
    nn.Module
):

    def __init__(
        self,
        dimension,
        maximum_length,
        base=10000.0,
    ):

        super().__init__()


        inverse_frequency = (

            1.0

            /

            (
                base
                **
                (
                    torch.arange(
                        0,
                        dimension,
                        2,
                    ).float()

                    /

                    dimension
                )
            )
        )


        positions = torch.arange(
            maximum_length
        ).float()


        frequencies = torch.outer(

            positions,

            inverse_frequency,
        )


        self.register_buffer(

            "cosine",

            frequencies.cos(),

            persistent=False,
        )


        self.register_buffer(

            "sine",

            frequencies.sin(),

            persistent=False,
        )


    def forward(
        self,
        Q,
        K,
    ):

        T = Q.shape[
            -2
        ]


        cosine = (

            self.cosine[
                :T
            ]

            .to(
                dtype=Q.dtype
            )

            [
                None,
                None,
                :,
                :
            ]
        )


        sine = (

            self.sine[
                :T
            ]

            .to(
                dtype=Q.dtype
            )

            [
                None,
                None,
                :,
                :
            ]
        )


        Q = apply_rotary(

            Q,

            cosine,

            sine,
        )


        K = apply_rotary(

            K,

            cosine,

            sine,
        )


        return Q, K


# ============================================================
# 13. SDPA CAUSAL SELF-ATTENTION
# ============================================================

class CausalSelfAttention(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()


        self.query = nn.Linear(

            d_model,

            d_model,

            bias=False,
        )


        self.key = nn.Linear(

            d_model,

            d_model,

            bias=False,
        )


        self.value = nn.Linear(

            d_model,

            d_model,

            bias=False,
        )


        self.output = nn.Linear(

            d_model,

            d_model,

            bias=False,
        )


        self.rotary = RotaryEmbedding(

            head_size,

            context_length,
        )


    def forward(
        self,
        x,
    ):

        B, T, C = x.shape


        Q = self.query(
            x
        )


        K = self.key(
            x
        )


        V = self.value(
            x
        )


        Q = (

            Q.reshape(

                B,
                T,
                number_of_heads,
                head_size,
            )

            .transpose(
                1,
                2,
            )
        )


        K = (

            K.reshape(

                B,
                T,
                number_of_heads,
                head_size,
            )

            .transpose(
                1,
                2,
            )
        )


        V = (

            V.reshape(

                B,
                T,
                number_of_heads,
                head_size,
            )

            .transpose(
                1,
                2,
            )
        )


        Q, K = self.rotary(
            Q,
            K,
        )


        out = F.scaled_dot_product_attention(

            Q,

            K,

            V,

            attn_mask=None,

            dropout_p=0.0,

            is_causal=True,
        )


        out = (

            out

            .transpose(
                1,
                2,
            )

            .contiguous()

            .reshape(

                B,
                T,
                d_model,
            )
        )


        return self.output(
            out
        )


# ============================================================
# 14. SWIGLU
# ============================================================

class SwiGLU(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()


        self.gate = nn.Linear(

            d_model,

            mlp_hidden,

            bias=False,
        )


        self.up = nn.Linear(

            d_model,

            mlp_hidden,

            bias=False,
        )


        self.down = nn.Linear(

            mlp_hidden,

            d_model,

            bias=False,
        )


    def forward(
        self,
        x,
    ):

        return self.down(

            F.silu(
                self.gate(x)
            )

            *

            self.up(x)
        )


# ============================================================
# 15. TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()


        self.attention_norm = RMSNorm(
            d_model
        )


        self.attention = (
            CausalSelfAttention()
        )


        self.mlp_norm = RMSNorm(
            d_model
        )


        self.mlp = (
            SwiGLU()
        )


    def forward(
        self,
        x,
    ):

        x = (

            x

            +

            self.attention(

                self.attention_norm(
                    x
                )
            )
        )


        x = (

            x

            +

            self.mlp(

                self.mlp_norm(
                    x
                )
            )
        )


        return x


# ============================================================
# 16. BLANK-1
# ============================================================

class BlankOne(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()


        self.token_embedding = nn.Embedding(

            vocab_size,

            d_model,
        )


        self.blocks = nn.ModuleList([

            TransformerBlock()

            for _ in range(
                number_of_layers
            )

        ])


        self.final_norm = RMSNorm(
            d_model
        )


        self.apply(
            self._initialize_weights
        )


        # ----------------------------------------------------
        # Residual projection scaling
        # ----------------------------------------------------

        residual_scale = (

            1.0

            /

            math.sqrt(

                2

                *

                number_of_layers
            )
        )


        for block in self.blocks:

            block.attention.output.weight.data.mul_(

                residual_scale
            )


            block.mlp.down.weight.data.mul_(

                residual_scale
            )


    def _initialize_weights(
        self,
        module,
    ):

        if isinstance(
            module,
            nn.Linear,
        ):

            nn.init.normal_(

                module.weight,

                mean=0.0,

                std=0.02,
            )


        elif isinstance(
            module,
            nn.Embedding,
        ):

            nn.init.normal_(

                module.weight,

                mean=0.0,

                std=0.02,
            )


    def forward(
        self,
        X,
        targets=None,
    ):

        B, T = X.shape


        if T > context_length:

            raise ValueError(
                "Input exceeds context length."
            )


        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        x = self.token_embedding(
            X
        )


        # ----------------------------------------------------
        # Transformer stack
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(
                x
            )


        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        x = self.final_norm(
            x
        )


        # ----------------------------------------------------
        # Weight-tied output projection
        # ----------------------------------------------------

        logits = F.linear(

            x,

            self.token_embedding.weight,
        )


        loss = None


        if targets is not None:

            loss = F.cross_entropy(

                logits.reshape(

                    -1,

                    vocab_size,
                ),

                targets.reshape(
                    -1
                ),
            )


        return logits, loss


# ============================================================
# 17. CREATE MODEL
# ============================================================

model = BlankOne().to(
    device
)


parameter_count = sum(

    p.numel()

    for p in model.parameters()
)


EXPECTED_PARAMETERS = (
    33_560_064
)


print(
    "\nTRAINABLE PARAMETERS:",
    f"{parameter_count:,}"
)


if (
    parameter_count
    !=
    EXPECTED_PARAMETERS
):

    raise RuntimeError(

        f"Expected "
        f"{EXPECTED_PARAMETERS:,} parameters, "
        f"got {parameter_count:,}"
    )


print(
    "Parameter-count check: OK"
)


# ============================================================
# 18. FUSED ADAMW
# ============================================================

print(
    "\nCreating fused AdamW optimizer..."
)


try:

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=maximum_learning_rate,

        betas=(
            0.9,
            0.95,
        ),

        weight_decay=weight_decay,

        fused=True,
    )


    OPTIMIZER_MODE = "fused"


except Exception as error:

    print(
        "Fused AdamW unavailable."
    )

    print(
        "Falling back to foreach AdamW."
    )

    print(
        "Reason:",
        error
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=maximum_learning_rate,

        betas=(
            0.9,
            0.95,
        ),

        weight_decay=weight_decay,

        foreach=True,
    )


    OPTIMIZER_MODE = "foreach"


print(
    "Optimizer mode:",
    OPTIMIZER_MODE
)


# ============================================================
# 19. BF16 AUTOCAST
# ============================================================

def bf16_context():

    return torch.autocast(

        device_type="cuda",

        dtype=torch.bfloat16,
    )


# ============================================================
# 20. LEARNING RATE
# ============================================================

def get_learning_rate(
    update_index,
):

    # --------------------------------------------------------
    # Linear warmup
    # --------------------------------------------------------

    if update_index < warmup_updates:

        return (

            maximum_learning_rate

            *

            (
                update_index
                +
                1
            )

            /

            warmup_updates
        )


    # --------------------------------------------------------
    # Cosine decay
    # --------------------------------------------------------

    progress = (

        update_index
        -
        warmup_updates

    ) / (

        training_updates
        -
        warmup_updates
    )


    progress = min(

        max(
            progress,
            0.0,
        ),

        1.0,
    )


    cosine = (

        0.5

        *

        (
            1.0

            +

            math.cos(

                math.pi

                *

                progress
            )
        )
    )


    return (

        minimum_learning_rate

        +

        cosine

        *

        (
            maximum_learning_rate

            -

            minimum_learning_rate
        )
    )


# ============================================================
# 21. EVALUATION
# ============================================================

@torch.inference_mode()
def evaluate(
    batches,
):

    model.eval()


    results = {}


    for split in [
        "train",
        "validation",
    ]:

        total_loss = 0.0


        for _ in range(
            batches
        ):

            X, Y = get_batch(
                split
            )


            with bf16_context():

                _, loss = model(
                    X,
                    Y,
                )


            total_loss += (
                loss.item()
            )


        average_loss = (

            total_loss

            /

            batches
        )


        perplexity = math.exp(

            min(
                average_loss,
                20.0,
            )
        )


        results[
            split
        ] = {

            "loss":
                average_loss,

            "perplexity":
                perplexity,
        }


    model.train()


    return results


# ============================================================
# 22. CHECKPOINT SAVING
# ============================================================

def save_checkpoint(
    filename,
    completed_updates,
    best_validation_loss,
):

    checkpoint = {

        "model_name":
            "Blank-1 v0.1",

        "completed_updates":
            completed_updates,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "best_validation_loss":
            best_validation_loss,

        "torch_rng_state":
            torch.get_rng_state(),

        "cuda_rng_state":
            torch.cuda.get_rng_state_all(),

        "config": {

            "vocab_size":
                vocab_size,

            "context_length":
                context_length,

            "batch_size":
                batch_size,

            "d_model":
                d_model,

            "layers":
                number_of_layers,

            "heads":
                number_of_heads,

            "head_size":
                head_size,

            "mlp_hidden":
                mlp_hidden,

            "parameters":
                parameter_count,

            "precision":
                "BF16 autocast",

            "attention":
                "SDPA",

            "optimizer":
                OPTIMIZER_MODE,

            "training_updates":
                training_updates,

            "token_predictions":
                actual_token_predictions,
        },
    }


    torch.save(

        checkpoint,

        filename,
    )


# ============================================================
# 23. RESUME
# ============================================================

completed_updates = 0


best_validation_loss = float(
    "inf"
)


if (
    RESUME
    and
    os.path.exists(
        LATEST_CHECKPOINT
    )
):

    print(
        "\nRESUMING EXISTING BLANK-1 RUN"
    )


    checkpoint = torch.load(

        LATEST_CHECKPOINT,

        map_location=device,

        weights_only=False,
    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )


    optimizer.load_state_dict(

        checkpoint[
            "optimizer_state_dict"
        ]
    )


    completed_updates = checkpoint[
        "completed_updates"
    ]


    best_validation_loss = checkpoint[
        "best_validation_loss"
    ]


    if "torch_rng_state" in checkpoint:

        torch.set_rng_state(

            checkpoint[
                "torch_rng_state"
            ]
        )


    if "cuda_rng_state" in checkpoint:

        torch.cuda.set_rng_state_all(

            checkpoint[
                "cuda_rng_state"
            ]
        )


    print(
        "Completed updates:",
        f"{completed_updates:,}"
    )


    print(
        "Remaining updates:",
        f"{training_updates - completed_updates:,}"
    )


# ============================================================
# 24. INITIAL EVALUATION
# ============================================================

if completed_updates == 0:

    print(
        "\nINITIAL EVALUATION"
    )


    initial_metrics = evaluate(
        evaluation_batches
    )


    initial_train_loss = (

        initial_metrics[
            "train"
        ][
            "loss"
        ]
    )


    initial_validation_loss = (

        initial_metrics[
            "validation"
        ][
            "loss"
        ]
    )


    initial_perplexity = (

        initial_metrics[
            "validation"
        ][
            "perplexity"
        ]
    )


    best_validation_loss = (
        initial_validation_loss
    )


    print(

        f"updates={0:5d} | "

        f"train={initial_train_loss:.4f} | "

        f"val={initial_validation_loss:.4f} | "

        f"ppl={initial_perplexity:.2f}"
    )


    save_checkpoint(

        BEST_CHECKPOINT,

        0,

        best_validation_loss,
    )


    save_checkpoint(

        LATEST_CHECKPOINT,

        0,

        best_validation_loss,
    )


# ============================================================
# 25. TRAINING LOOP
# ============================================================

print(
    "\nTRAINING BLANK-1 v0.1\n"
)


model.train()


torch.cuda.reset_peak_memory_stats(
    0
)


session_start = (
    time.perf_counter()
)


report_start = (
    time.perf_counter()
)


updates_since_report = 0


while (
    completed_updates
    <
    training_updates
):

    update_index = (
        completed_updates
    )


    # ========================================================
    # LR
    # ========================================================

    learning_rate = (
        get_learning_rate(
            update_index
        )
    )


    for parameter_group in (
        optimizer.param_groups
    ):

        parameter_group[
            "lr"
        ] = learning_rate


    # ========================================================
    # BATCH
    # ========================================================

    X, Y = get_batch(
        "train"
    )


    # ========================================================
    # ZERO GRAD
    # ========================================================

    optimizer.zero_grad(
        set_to_none=True
    )


    # ========================================================
    # BF16 FORWARD
    # ========================================================

    with bf16_context():

        _, loss = model(
            X,
            Y,
        )


    if not torch.isfinite(
        loss
    ):

        raise RuntimeError(

            f"Non-finite loss at update "
            f"{completed_updates + 1}"
        )


    # ========================================================
    # BACKWARD
    # ========================================================

    loss.backward()


    # ========================================================
    # GRADIENT CLIP
    # ========================================================

    gradient_norm = (
        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            gradient_clip,
        )
    )


    if not torch.isfinite(
        gradient_norm
    ):

        raise RuntimeError(

            f"Non-finite gradient at update "
            f"{completed_updates + 1}"
        )


    # ========================================================
    # UPDATE
    # ========================================================

    optimizer.step()


    completed_updates += 1

    updates_since_report += 1


    # ========================================================
    # EVALUATION
    # ========================================================

    if (
        completed_updates
        %
        evaluation_interval
        ==
        0
    ):

        torch.cuda.synchronize()


        now = time.perf_counter()


        training_segment_time = (

            now

            -

            report_start
        )


        processed_tokens = (

            updates_since_report

            *

            tokens_per_update
        )


        training_throughput = (

            processed_tokens

            /

            training_segment_time
        )


        metrics = evaluate(
            evaluation_batches
        )


        train_loss = (

            metrics[
                "train"
            ][
                "loss"
            ]
        )


        validation_loss = (

            metrics[
                "validation"
            ][
                "loss"
            ]
        )


        validation_perplexity = (

            metrics[
                "validation"
            ][
                "perplexity"
        ]
        )


        dataset_equivalent = (

            completed_updates

            *

            tokens_per_update

            /

            train_tokens
        )


        reserved_vram = (

            torch.cuda.memory_reserved(
                0
            )

            /

            1024**3
        )


        print(

            f"updates={completed_updates:5d} | "

            f"train={train_loss:.4f} | "

            f"val={validation_loss:.4f} | "

            f"ppl={validation_perplexity:.2f} | "

            f"lr={learning_rate:.2e} | "

            f"grad={gradient_norm.item():.3f} | "

            f"tok/s={training_throughput:,.0f} | "

            f"data={dataset_equivalent:.3f}x | "

            f"VRAM={reserved_vram:.2f} GB"
        )


        # ====================================================
        # BEST CHECKPOINT
        # ====================================================

        if (
            validation_loss
            <
            best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )


            save_checkpoint(

                BEST_CHECKPOINT,

                completed_updates,

                best_validation_loss,
            )


            print(
                "  -> new best checkpoint"
            )


        # ====================================================
        # LATEST CHECKPOINT
        # ====================================================

        save_checkpoint(

            LATEST_CHECKPOINT,

            completed_updates,

            best_validation_loss,
        )


        report_start = (
            time.perf_counter()
        )


        updates_since_report = 0


# ============================================================
# 26. FINAL EVALUATION
# ============================================================

torch.cuda.synchronize()


training_end = (
    time.perf_counter()
)


print(
    "\nRUNNING FINAL EVALUATION..."
)


final_metrics = evaluate(
    final_evaluation_batches
)


final_train_loss = (

    final_metrics[
        "train"
    ][
        "loss"
    ]
)


final_validation_loss = (

    final_metrics[
        "validation"
    ][
        "loss"
    ]
)


final_perplexity = (

    final_metrics[
        "validation"
    ][
        "perplexity"
    ]
)


print(
    "\n"
    +
    "=" * 70
)

print(
    "FINAL RESULTS"
)

print(
    "=" * 70
)


print(
    "Completed updates:",
    f"{completed_updates:,}"
)


print(
    "Token predictions:",
    f"{completed_updates * tokens_per_update:,}"
)


print(
    "Dataset equivalents:",
    round(

        (
            completed_updates

            *

            tokens_per_update
        )

        /

        train_tokens,

        4,
    )
)


print(
    "Training loss:",
    round(
        final_train_loss,
        4,
    )
)


print(
    "Validation loss:",
    round(
        final_validation_loss,
        4,
    )
)


print(
    "Validation perplexity:",
    round(
        final_perplexity,
        2,
    )
)


# ============================================================
# 27. FINAL BEST CHECK
# ============================================================

if (
    final_validation_loss
    <
    best_validation_loss
):

    best_validation_loss = (
        final_validation_loss
    )


    save_checkpoint(

        BEST_CHECKPOINT,

        completed_updates,

        best_validation_loss,
    )


    print(
        "Final model became the new best checkpoint."
    )


print(
    "Best validation loss:",
    round(
        best_validation_loss,
        4,
    )
)


# ============================================================
# 28. BLANK-0 COMPARISON
# ============================================================

difference_vs_blank0 = (

    best_validation_loss

    -

    BLANK0_BEST_VALIDATION
)


print(
    "\nBLANK-0 COMPARISON"
)


print(
    "Blank-0 best validation:",
    round(
        BLANK0_BEST_VALIDATION,
        4,
    )
)


print(
    "Blank-1 best validation:",
    round(
        best_validation_loss,
        4,
    )
)


print(
    "Difference:",
    f"{difference_vs_blank0:+.4f}"
)


# ============================================================
# 29. SAVE FINAL / LATEST
# ============================================================

save_checkpoint(

    FINAL_CHECKPOINT,

    completed_updates,

    best_validation_loss,
)


save_checkpoint(

    LATEST_CHECKPOINT,

    completed_updates,

    best_validation_loss,
)


print(
    "\nFinal checkpoint:",
    FINAL_CHECKPOINT
)

print(
    "Best checkpoint:",
    BEST_CHECKPOINT
)

print(
    "Latest checkpoint:",
    LATEST_CHECKPOINT
)


# ============================================================
# 30. PERFORMANCE REPORT
# ============================================================

session_seconds = (

    training_end

    -

    session_start
)


total_training_predictions = (

    completed_updates

    *

    tokens_per_update
)


average_throughput = (

    total_training_predictions

    /

    session_seconds
)


print(
    "\nPERFORMANCE"
)


print(
    "Training session time:",
    round(
        session_seconds,
        2,
    ),
    "seconds"
)


print(
    "Training session time:",
    round(
        session_seconds
        /
        60,
        2,
    ),
    "minutes"
)


print(
    "Average overall throughput:",
    f"{average_throughput:,.0f}",
    "tokens/sec"
)


print(
    "Peak allocated VRAM:",
    round(
        torch.cuda.max_memory_allocated(
            0
        )
        /
        1024**3,
        2,
    ),
    "GB"
)


print(
    "Peak reserved VRAM:",
    round(
        torch.cuda.max_memory_reserved(
            0
        )
        /
        1024**3,
        2,
    ),
    "GB"
)


# ============================================================
# 31. LOAD TRUE BEST MODEL
# ============================================================

best_checkpoint = torch.load(

    BEST_CHECKPOINT,

    map_location=device,

    weights_only=False,
)


model.load_state_dict(

    best_checkpoint[
        "model_state_dict"
    ]
)


print(
    "\nLoaded true best Blank-1 checkpoint."
)


print(
    "Best checkpoint update:",
    best_checkpoint[
        "completed_updates"
    ]
)


print(
    "Best checkpoint validation loss:",
    best_checkpoint[
        "best_validation_loss"
    ]
)


# ============================================================
# 32. GENERATION
# ============================================================

@torch.inference_mode()
def generate(

    prompt,

    number_of_tokens=250,

    temperature=0.8,

    top_k=50,
):

    model.eval()


    prompt_ids = (

        tokenizer

        .encode(
            prompt
        )

        .ids
    )


    generated = torch.tensor(

        [
            prompt_ids
        ],

        dtype=torch.long,

        device=device,
    )


    for _ in range(
        number_of_tokens
    ):

        context = generated[
            :,
            -context_length:
        ]


        with bf16_context():

            logits, _ = model(
                context
            )


        # Sampling math back in FP32.

        logits = (

            logits[
                :,
                -1,
                :
            ]

            .float()

            /

            temperature
        )


        if (
            top_k is not None
            and
            top_k < vocab_size
        ):

            top_values, _ = torch.topk(

                logits,

                top_k,
            )


            cutoff = (

                top_values[
                    :,
                    -1
                ]

                .unsqueeze(
                    -1
                )
            )


            logits = logits.masked_fill(

                logits < cutoff,

                float("-inf"),
            )


        probabilities = F.softmax(

            logits,

            dim=-1,
        )


        next_token = torch.multinomial(

            probabilities,

            num_samples=1,
        )


        generated = torch.cat(

            [
                generated,
                next_token,
            ],

            dim=1,
        )


    return tokenizer.decode(

        generated[
            0
        ].tolist()
    )


# ============================================================
# 33. GENERATION TESTS
# ============================================================

prompts = [

    "Artificial intelligence is",

    "The history of Europe",

    "A computer network is",

    "The human brain",

    "In mathematics, a function",
]


for index, prompt in enumerate(
    prompts,
    start=1,
):

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        f"GENERATED SAMPLE {index}"
    )

    print(
        "=" * 70
        +
        "\n"
    )


    print(

        generate(

            prompt,

            number_of_tokens=250,

            temperature=0.8,

            top_k=50,
        )
    )


print(
    "\nDONE"
)