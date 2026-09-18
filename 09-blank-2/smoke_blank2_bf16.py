import gc
import json
import math
import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. FILES
# ============================================================

DATA_FILE = "blank_dataset_v3_tokens.bin"
METADATA_FILE = "blank_dataset_v3_tokens_meta.json"


# ============================================================
# 2. ARCHITECTURE
# ============================================================

CONTEXT = 512

D_MODEL = 640
LAYERS = 12

HEADS = 10
HEAD_DIM = 64

MLP_HIDDEN = 1707

EXPECTED_PARAMETERS = 61_627_520


# ============================================================
# 3. PERFORMANCE TEST
# ============================================================

BATCH_CANDIDATES = [
    8,
    12,
    16,
    20,
    24,
    32,
]

BENCHMARK_WARMUP = 5
BENCHMARK_UPDATES = 15


# ============================================================
# 4. SMOKE TEST
# ============================================================

SMOKE_UPDATES = 300
SMOKE_WARMUP = 100

MAX_LR = 2.0e-4

WEIGHT_DECAY = 0.1
GRADIENT_CLIP = 1.0

EVAL_BATCHES = 8


# ============================================================
# 5. VERIFY FILES
# ============================================================

for filename in [
    DATA_FILE,
    METADATA_FILE,
]:

    if not os.path.exists(filename):

        raise FileNotFoundError(
            filename
        )


# ============================================================
# 6. METADATA
# ============================================================

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8",
) as f:

    metadata = json.load(f)


VOCAB = metadata["vocab_size"]

TOTAL_TOKENS = metadata["total_tokens"]

TRAIN_TOKENS = metadata["train_tokens"]

VALIDATION_TOKENS = metadata["validation_tokens"]


print("=" * 72)
print("BLANK-2 BF16 SMOKE + SPEED AUTOTUNE")
print("=" * 72)

print()
print(
    "Total tokens:",
    f"{TOTAL_TOKENS:,}",
)

print(
    "Training tokens:",
    f"{TRAIN_TOKENS:,}",
)

print(
    "Validation tokens:",
    f"{VALIDATION_TOKENS:,}",
)

print(
    "Vocabulary:",
    f"{VOCAB:,}",
)


# ============================================================
# 7. GPU
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "ROCm GPU unavailable."
    )


device = torch.device(
    "cuda:0"
)


print()
print(
    "GPU:",
    torch.cuda.get_device_name(0),
)

print(
    "PyTorch:",
    torch.__version__,
)

print(
    "HIP:",
    torch.version.hip,
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
# 8. MODEL CONFIG
# ============================================================

assert (
    D_MODEL
    ==
    HEADS * HEAD_DIM
)

assert (
    HEAD_DIM % 2 == 0
)


print()
print("BLANK-2 ARCHITECTURE")

print(
    "Context:",
    CONTEXT,
)

print(
    "Width:",
    D_MODEL,
)

print(
    "Layers:",
    LAYERS,
)

print(
    "Heads:",
    HEADS,
)

print(
    "Head dimension:",
    HEAD_DIM,
)

print(
    "SwiGLU hidden:",
    MLP_HIDDEN,
)

print(
    "Precision: BF16 autocast",
)

print(
    "Attention: SDPA",
)

print(
    "Optimizer: fused AdamW",
)


# ============================================================
# 9. DATASET
# ============================================================
#
# Keep Dataset v3 as INT32 on the GPU.
#
# 875.8M tokens * 4 bytes ≈ 3.26 GiB.
#
# Only each selected mini-batch is converted to INT64.
# ============================================================

print()
print("Loading Dataset v3...")

disk_tokens = np.memmap(

    DATA_FILE,

    dtype=np.uint16,

    mode="r",
)


if len(disk_tokens) != TOTAL_TOKENS:

    raise RuntimeError(
        "Dataset length does not match metadata."
    )


print(
    "Converting corpus to INT32..."
)


cpu_array = np.asarray(

    disk_tokens,

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
del disk_tokens

gc.collect()


train_data = all_tokens[
    :TRAIN_TOKENS
]


validation_data = all_tokens[
    TRAIN_TOKENS:
]


dataset_gpu_gb = (

    all_tokens.numel()

    *

    all_tokens.element_size()

    /

    1024**3
)


print(
    "Dataset GPU memory:",
    round(
        dataset_gpu_gb,
        2,
    ),
    "GB",
)


# ============================================================
# 10. BATCH SAMPLER
# ============================================================

offsets = torch.arange(

    CONTEXT,

    device=device,
)


def get_batch(
    split,
    batch_size,
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
            CONTEXT
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

        offsets[
            None,
            :
        ]
    )


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


        return (

            x

            *

            torch.rsqrt(

                mean_square

                +

                self.epsilon
            )

            *

            self.weight
        )


# ============================================================
# 12. ROPE
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


    rotated_even = (

        even * cosine

        -

        odd * sine
    )


    rotated_odd = (

        even * sine

        +

        odd * cosine
    )


    return torch.stack(

        [
            rotated_even,
            rotated_odd,
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
# 13. SDPA ATTENTION
# ============================================================

class CausalSelfAttention(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()


        self.query = nn.Linear(

            D_MODEL,

            D_MODEL,

            bias=False,
        )


        self.key = nn.Linear(

            D_MODEL,

            D_MODEL,

            bias=False,
        )


        self.value = nn.Linear(

            D_MODEL,

            D_MODEL,

            bias=False,
        )


        self.output = nn.Linear(

            D_MODEL,

            D_MODEL,

            bias=False,
        )


        self.rotary = RotaryEmbedding(

            HEAD_DIM,

            CONTEXT,
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
                HEADS,
                HEAD_DIM,
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
                HEADS,
                HEAD_DIM,
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
                HEADS,
                HEAD_DIM,
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
                D_MODEL,
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

            D_MODEL,

            MLP_HIDDEN,

            bias=False,
        )


        self.up = nn.Linear(

            D_MODEL,

            MLP_HIDDEN,

            bias=False,
        )


        self.down = nn.Linear(

            MLP_HIDDEN,

            D_MODEL,

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
            D_MODEL
        )


        self.attention = (
            CausalSelfAttention()
        )


        self.mlp_norm = RMSNorm(
            D_MODEL
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
# 16. BLANK-2
# ============================================================

class BlankTwo(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()


        self.token_embedding = nn.Embedding(

            VOCAB,

            D_MODEL,
        )


        self.blocks = nn.ModuleList([

            TransformerBlock()

            for _ in range(
                LAYERS
            )

        ])


        self.final_norm = RMSNorm(
            D_MODEL
        )


        self.apply(
            self._initialize_weights
        )


        # ----------------------------------------
        # Residual output scaling
        # ----------------------------------------

        residual_scale = (

            1.0

            /

            math.sqrt(
                2
                *
                LAYERS
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

        x = self.token_embedding(
            X
        )


        for block in self.blocks:

            x = block(
                x
            )


        x = self.final_norm(
            x
        )


        # Weight-tied output projection.

        logits = F.linear(

            x,

            self.token_embedding.weight,
        )


        loss = None


        if targets is not None:

            loss = F.cross_entropy(

                logits.reshape(

                    -1,

                    VOCAB,
                ),

                targets.reshape(
                    -1
                ),
            )


        return logits, loss


# ============================================================
# 17. BF16
# ============================================================

def bf16_context():

    return torch.autocast(

        device_type="cuda",

        dtype=torch.bfloat16,
    )


# ============================================================
# 18. OPTIMIZER
# ============================================================

def create_optimizer(
    model,
):

    try:

        optimizer = torch.optim.AdamW(

            model.parameters(),

            lr=MAX_LR,

            betas=(
                0.9,
                0.95,
            ),

            weight_decay=WEIGHT_DECAY,

            fused=True,
        )


        return optimizer, "fused"


    except Exception:

        optimizer = torch.optim.AdamW(

            model.parameters(),

            lr=MAX_LR,

            betas=(
                0.9,
                0.95,
            ),

            weight_decay=WEIGHT_DECAY,

            foreach=True,
        )


        return optimizer, "foreach"


# ============================================================
# 19. CLEANUP
# ============================================================

def cleanup():

    gc.collect()

    torch.cuda.empty_cache()

    torch.cuda.synchronize()


# ============================================================
# 20. PARAMETER CHECK
# ============================================================

print()
print("Checking Blank-2 parameter count...")


torch.manual_seed(
    42
)


temporary_model = BlankTwo().to(
    device
)


parameter_count = sum(

    p.numel()

    for p in temporary_model.parameters()
)


print(
    "TRAINABLE PARAMETERS:",
    f"{parameter_count:,}",
)


if parameter_count != EXPECTED_PARAMETERS:

    raise RuntimeError(

        f"Expected {EXPECTED_PARAMETERS:,} "
        f"parameters, got {parameter_count:,}"
    )


print(
    "Parameter-count check: OK"
)


del temporary_model

cleanup()


# ============================================================
# 21. TRAINING UPDATE
# ============================================================

def train_update(
    model,
    optimizer,
    batch_size,
):

    X, Y = get_batch(

        "train",

        batch_size,
    )


    optimizer.zero_grad(
        set_to_none=True
    )


    with bf16_context():

        _, loss = model(
            X,
            Y,
        )


    if not torch.isfinite(
        loss
    ):

        raise RuntimeError(
            "Non-finite loss."
        )


    loss.backward()


    gradient_norm = (
        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            GRADIENT_CLIP,
        )
    )


    if not torch.isfinite(
        gradient_norm
    ):

        raise RuntimeError(
            "Non-finite gradient norm."
        )


    optimizer.step()


    return (
        loss.item(),
        gradient_norm.item(),
    )


# ============================================================
# 22. SPEED BENCHMARK
# ============================================================

def benchmark_batch(
    batch_size,
):

    torch.manual_seed(
        123
    )

    torch.cuda.manual_seed_all(
        123
    )


    model = BlankTwo().to(
        device
    )


    optimizer, optimizer_mode = (
        create_optimizer(
            model
        )
    )


    # ----------------------------------------
    # Warmup
    # ----------------------------------------

    for _ in range(
        BENCHMARK_WARMUP
    ):

        train_update(

            model,

            optimizer,

            batch_size,
        )


    torch.cuda.synchronize()


    torch.cuda.reset_peak_memory_stats(
        0
    )


    # ----------------------------------------
    # Timed updates
    # ----------------------------------------

    start = time.perf_counter()


    last_loss = None


    for _ in range(
        BENCHMARK_UPDATES
    ):

        last_loss, _ = train_update(

            model,

            optimizer,

            batch_size,
        )


    torch.cuda.synchronize()


    elapsed = (

        time.perf_counter()

        -

        start
    )


    tokens_per_update = (

        batch_size

        *

        CONTEXT
    )


    total_tokens = (

        BENCHMARK_UPDATES

        *

        tokens_per_update
    )


    throughput = (

        total_tokens

        /

        elapsed
    )


    peak_allocated = (

        torch.cuda.max_memory_allocated(
            0
        )

        /

        1024**3
    )


    peak_reserved = (

        torch.cuda.max_memory_reserved(
            0
        )

        /

        1024**3
    )


    result = {

        "batch":
            batch_size,

        "tokens_per_update":
            tokens_per_update,

        "tokens_per_second":
            throughput,

        "milliseconds_per_update":
            (
                elapsed
                /
                BENCHMARK_UPDATES
                *
                1000
            ),

        "peak_allocated_gb":
            peak_allocated,

        "peak_reserved_gb":
            peak_reserved,

        "last_loss":
            last_loss,

        "optimizer":
            optimizer_mode,
    }


    del optimizer
    del model

    cleanup()


    return result


# ============================================================
# 23. BATCH AUTOTUNE
# ============================================================

print()
print("=" * 72)
print("PHASE 1 — BATCH SIZE SPEED TEST")
print("=" * 72)


benchmark_results = []


for batch_size in BATCH_CANDIDATES:

    print()
    print(
        f"Testing batch {batch_size}..."
    )


    try:

        result = benchmark_batch(
            batch_size
        )


        benchmark_results.append(
            result
        )


        print(
            "Tokens/update:",
            f"{result['tokens_per_update']:,}",
        )


        print(
            "Tokens/sec:",
            f"{result['tokens_per_second']:,.0f}",
        )


        print(
            "ms/update:",
            f"{result['milliseconds_per_update']:.2f}",
        )


        print(
            "Peak allocated:",
            f"{result['peak_allocated_gb']:.2f} GB",
        )


        print(
            "Peak reserved:",
            f"{result['peak_reserved_gb']:.2f} GB",
        )


        print(
            "Loss:",
            f"{result['last_loss']:.4f}",
        )


    except torch.cuda.OutOfMemoryError:

        print(
            "OUT OF MEMORY."
        )

        cleanup()

        break


    except Exception as error:

        print(
            "FAILED:",
            type(error).__name__,
            error,
        )

        cleanup()


if not benchmark_results:

    raise RuntimeError(
        "No batch size completed successfully."
    )


# User explicitly wants maximum full-training speed,
# so take the absolute throughput winner.

winner = max(

    benchmark_results,

    key=lambda x:
        x["tokens_per_second"],
)


BEST_BATCH = winner[
    "batch"
]


BEST_SPEED = winner[
    "tokens_per_second"
]


TOKENS_PER_UPDATE = (

    BEST_BATCH

    *

    CONTEXT
)


FULL_TRAINING_UPDATES = math.ceil(

    TRAIN_TOKENS

    /

    TOKENS_PER_UPDATE
)


FULL_TOKEN_PREDICTIONS = (

    FULL_TRAINING_UPDATES

    *

    TOKENS_PER_UPDATE
)


raw_training_seconds = (

    FULL_TOKEN_PREDICTIONS

    /

    BEST_SPEED
)


print()
print("=" * 72)
print("BATCH AUTOTUNE WINNER")
print("=" * 72)

print(
    "Batch:",
    BEST_BATCH,
)

print(
    "Measured speed:",
    f"{BEST_SPEED:,.0f} tokens/sec",
)

print(
    "Tokens/update:",
    f"{TOKENS_PER_UPDATE:,}",
)

print(
    "Full-run updates:",
    f"{FULL_TRAINING_UPDATES:,}",
)

print(
    "Full token predictions:",
    f"{FULL_TOKEN_PREDICTIONS:,}",
)

print(
    "Dataset equivalents:",
    f"{FULL_TOKEN_PREDICTIONS / TRAIN_TOKENS:.6f}",
)

print(
    "Raw full-run estimate:",
    f"{raw_training_seconds / 60:.2f} minutes",
)


# ============================================================
# 24. FRESH MODEL FOR HEALTH TEST
# ============================================================

print()
print("=" * 72)
print("PHASE 2 — 300-UPDATE HEALTH TEST")
print("=" * 72)


cleanup()


torch.manual_seed(
    42
)

torch.cuda.manual_seed_all(
    42
)


model = BlankTwo().to(
    device
)


optimizer, optimizer_mode = (
    create_optimizer(
        model
    )
)


print()
print(
    "Batch:",
    BEST_BATCH,
)

print(
    "Optimizer:",
    optimizer_mode,
)

print(
    "Smoke updates:",
    SMOKE_UPDATES,
)

print(
    "Smoke token predictions:",
    f"{SMOKE_UPDATES * TOKENS_PER_UPDATE:,}",
)


# ============================================================
# 25. SMOKE LEARNING RATE
# ============================================================

def smoke_learning_rate(
    completed_updates,
):

    if completed_updates < SMOKE_WARMUP:

        return (

            MAX_LR

            *

            (
                completed_updates
                +
                1
            )

            /

            SMOKE_WARMUP
        )


    return MAX_LR


# ============================================================
# 26. EVALUATION
# ============================================================

@torch.inference_mode()
def evaluate():

    model.eval()


    results = {}


    for split in [
        "train",
        "validation",
    ]:

        losses = []


        for _ in range(
            EVAL_BATCHES
        ):

            X, Y = get_batch(

                split,

                BEST_BATCH,
            )


            with bf16_context():

                _, loss = model(
                    X,
                    Y,
                )


            losses.append(
                loss.item()
            )


        results[
            split
        ] = (

            sum(losses)

            /

            len(losses)
        )


    model.train()


    return results


# ============================================================
# 27. INITIAL EVALUATION
# ============================================================

print()
print("INITIAL EVALUATION")


initial_metrics = evaluate()


initial_train_loss = (
    initial_metrics["train"]
)


initial_validation_loss = (
    initial_metrics["validation"]
)


print(
    "Expected random CE ≈ ln(vocab):",
    f"{math.log(VOCAB):.4f}",
)

print(
    "Train loss:",
    f"{initial_train_loss:.4f}",
)

print(
    "Validation loss:",
    f"{initial_validation_loss:.4f}",
)


# ============================================================
# 28. SMOKE TRAINING
# ============================================================

print()
print("TRAINING SMOKE TEST")
print()


torch.cuda.reset_peak_memory_stats(
    0
)


smoke_start = (
    time.perf_counter()
)


segment_start = (
    smoke_start
)


last_gradient_norm = None


for completed_updates in range(
    1,
    SMOKE_UPDATES + 1,
):

    learning_rate = (
        smoke_learning_rate(
            completed_updates - 1
        )
    )


    for group in (
        optimizer.param_groups
    ):

        group["lr"] = (
            learning_rate
        )


    batch_loss, gradient_norm = (
        train_update(

            model,

            optimizer,

            BEST_BATCH,
        )
    )


    last_gradient_norm = (
        gradient_norm
    )


    if (
        completed_updates % 100
        ==
        0
    ):

        torch.cuda.synchronize()


        now = time.perf_counter()


        elapsed_segment = (

            now

            -

            segment_start
        )


        segment_tokens = (

            100

            *

            TOKENS_PER_UPDATE
        )


        throughput = (

            segment_tokens

            /

            elapsed_segment
        )


        metrics = evaluate()


        reserved = (

            torch.cuda.memory_reserved(
                0
            )

            /

            1024**3
        )


        print(

            f"updates={completed_updates:3d} | "

            f"train={metrics['train']:.4f} | "

            f"val={metrics['validation']:.4f} | "

            f"batch_loss={batch_loss:.4f} | "

            f"grad={gradient_norm:.3f} | "

            f"lr={learning_rate:.2e} | "

            f"tok/s={throughput:,.0f} | "

            f"VRAM={reserved:.2f} GB"
        )


        segment_start = (
            time.perf_counter()
        )


# ============================================================
# 29. FINAL EVALUATION
# ============================================================

final_metrics = evaluate()


torch.cuda.synchronize()


smoke_end = (
    time.perf_counter()
)


final_train_loss = (
    final_metrics["train"]
)


final_validation_loss = (
    final_metrics["validation"]
)


finite_final = (

    math.isfinite(
        final_train_loss
    )

    and

    math.isfinite(
        final_validation_loss
    )
)


validation_decreased = (

    final_validation_loss

    <
    initial_validation_loss
)


peak_allocated = (

    torch.cuda.max_memory_allocated(
        0
    )

    /

    1024**3
)


peak_reserved = (

    torch.cuda.max_memory_reserved(
        0
    )

    /

    1024**3
)


# ============================================================
# 30. FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("BLANK-2 SMOKE TEST COMPLETE")
print("=" * 80)


print(
    "Parameters:",
    f"{parameter_count:,}",
)

print(
    "Selected batch:",
    BEST_BATCH,
)

print(
    "Tokens/update:",
    f"{TOKENS_PER_UPDATE:,}",
)

print(
    "Smoke updates:",
    f"{SMOKE_UPDATES:,}",
)

print(
    "Smoke token predictions:",
    f"{SMOKE_UPDATES * TOKENS_PER_UPDATE:,}",
)


print()
print(
    "Initial train loss:",
    round(
        initial_train_loss,
        4,
    ),
)

print(
    "Initial validation loss:",
    round(
        initial_validation_loss,
        4,
    ),
)

print(
    "Final train loss:",
    round(
        final_train_loss,
        4,
    ),
)

print(
    "Final validation loss:",
    round(
        final_validation_loss,
        4,
    ),
)


print()
print(
    "Smoke session time:",
    round(
        smoke_end
        -
        smoke_start,
        2,
    ),
    "seconds",
)

print(
    "Peak allocated VRAM:",
    round(
        peak_allocated,
        2,
    ),
    "GB",
)

print(
    "Peak reserved VRAM:",
    round(
        peak_reserved,
        2,
    ),
    "GB",
)


print()
print("HEALTH CHECK")

print(
    "Finite final loss:",
    finite_final,
)

print(
    "Validation loss decreased:",
    validation_decreased,
)

print(
    "Final gradient norm finite:",
    math.isfinite(
        last_gradient_norm
    ),
)


print()
print("=" * 80)
print("FULL BLANK-2 TRAINING ESTIMATE")
print("=" * 80)


print(
    "Training tokens:",
    f"{TRAIN_TOKENS:,}",
)

print(
    "Selected batch:",
    BEST_BATCH,
)

print(
    "Measured benchmark speed:",
    f"{BEST_SPEED:,.0f} tokens/sec",
)

print(
    "Required updates:",
    f"{FULL_TRAINING_UPDATES:,}",
)

print(
    "Token predictions:",
    f"{FULL_TOKEN_PREDICTIONS:,}",
)

print(
    "Raw compute estimate:",
    f"{raw_training_seconds / 60:.2f} minutes",
)

print(
    "Approx. +8% overhead:",
    f"{raw_training_seconds * 1.08 / 60:.2f} minutes",
)


if (
    finite_final
    and
    validation_decreased
):

    print()
    print(
        "RESULT: BLANK-2 BF16 TRAINING LOOKS HEALTHY."
    )

else:

    print()
    print(
        "RESULT: HEALTH CHECK FAILED."
    )


print()
print(
    "No checkpoint was saved."
)

print(
    "This was only the Blank-2 smoke/performance test."
)

print()
print("DONE")