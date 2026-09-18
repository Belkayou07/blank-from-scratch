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

DATA_FILE = "blank_dataset_v2_tokens.bin"

METADATA_FILE = "blank_dataset_v2_tokens_meta.json"

CHECKPOINT_FILE = "blank0_v02_best.pt"


for filename in [
    DATA_FILE,
    METADATA_FILE,
    CHECKPOINT_FILE
]:
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)


# ============================================================
# 2. GPU
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "ROCm GPU is unavailable."
    )


device = torch.device(
    "cuda:0"
)


print(
    "GPU:",
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
        2
    ),
    "GB"
)


# ============================================================
# 3. SDPA STATUS
# ============================================================

print(
    "\nSDPA BACKENDS ENABLED"
)


try:
    print(
        "Flash:",
        torch.backends.cuda.flash_sdp_enabled()
    )

    print(
        "Memory efficient:",
        torch.backends.cuda.mem_efficient_sdp_enabled()
    )

    print(
        "Math:",
        torch.backends.cuda.math_sdp_enabled()
    )

except Exception as error:

    print(
        "Could not query SDPA flags:",
        error
    )


# ============================================================
# 4. LOAD METADATA
# ============================================================

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
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


# ============================================================
# 5. SAME BLANK-0 ARCHITECTURE
# ============================================================

context_length = 512

batch_size = 16

d_model = 384

number_of_layers = 8

number_of_heads = 6

head_size = (
    d_model
    //
    number_of_heads
)

mlp_hidden = 1024


assert (
    d_model
    %
    number_of_heads
    == 0
)


assert (
    head_size
    %
    2
    == 0
)


TOKENS_PER_UPDATE = (
    batch_size
    *
    context_length
)


# ============================================================
# 6. BENCHMARK CONFIGURATION
# ============================================================

WARMUP_UPDATES = 10

TIMED_UPDATES = 50


TOTAL_BENCHMARK_UPDATES = (
    WARMUP_UPDATES
    +
    TIMED_UPDATES
)


print(
    "\nBENCHMARK CONFIGURATION"
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
    f"{TOKENS_PER_UPDATE:,}"
)

print(
    "Warmup updates:",
    WARMUP_UPDATES
)

print(
    "Timed updates:",
    TIMED_UPDATES
)


# ============================================================
# 7. LOAD DATASET ONTO GPU
# ============================================================

print(
    "\nLoading token dataset..."
)


binary_tokens = np.memmap(

    DATA_FILE,

    dtype=np.uint16,

    mode="r"
)


if len(binary_tokens) != total_tokens:

    raise RuntimeError(
        "Dataset token count mismatch."
    )


print(
    "Converting dataset to int64..."
)


cpu_array = np.asarray(

    binary_tokens,

    dtype=np.int64
)


cpu_tokens = torch.from_numpy(
    cpu_array
)


print(
    "Moving dataset to GPU..."
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


print(
    "Dataset GPU memory:",
    round(
        all_tokens.numel()
        *
        all_tokens.element_size()
        /
        1024**3,
        2
    ),
    "GB"
)


# ============================================================
# 8. PRECOMPUTE IDENTICAL RANDOM BATCH LOCATIONS
# ============================================================
#
# Every benchmark variant receives exactly the same sequence
# positions. That removes batch-selection randomness from the
# comparison.

cpu_generator = torch.Generator()

cpu_generator.manual_seed(
    123456
)


all_start_positions = torch.randint(

    low=0,

    high=(
        train_tokens
        -
        context_length
        -
        1
    ),

    size=(
        TOTAL_BENCHMARK_UPDATES + 1,
        batch_size
    ),

    generator=cpu_generator,

    dtype=torch.long
)


all_start_positions = (
    all_start_positions.to(
        device
    )
)


offsets = torch.arange(

    context_length,

    device=device
)


def make_batch(
    starts
):

    positions = (

        starts[:, None]

        +

        offsets[
            None,
            :
        ]
    )


    X = train_data[
        positions
    ]


    Y = train_data[
        positions + 1
    ]


    return X, Y


# ============================================================
# 9. RMSNORM
# ============================================================

class RMSNorm(
    nn.Module
):

    def __init__(
        self,
        dimension,
        epsilon=1e-5
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
        x
    ):

        mean_square = torch.mean(

            x * x,

            dim=-1,

            keepdim=True
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
# 10. RoPE
# ============================================================

def apply_rotary(
    x,
    cosine,
    sine
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
            rotated_odd
        ],

        dim=-1

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
        base=10000.0
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
                        2
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

            inverse_frequency
        )


        self.register_buffer(

            "cosine",

            frequencies.cos(),

            persistent=False
        )


        self.register_buffer(

            "sine",

            frequencies.sin(),

            persistent=False
        )


    def forward(
        self,
        Q,
        K
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
            sine
        )


        K = apply_rotary(

            K,
            cosine,
            sine
        )


        return Q, K


# ============================================================
# 11. ATTENTION
# ============================================================

class CausalSelfAttention(
    nn.Module
):

    def __init__(
        self,
        attention_mode
    ):

        super().__init__()


        self.attention_mode = (
            attention_mode
        )


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


        self.rotary = RotaryEmbedding(

            head_size,

            context_length
        )


        if (
            attention_mode
            ==
            "manual"
        ):

            causal_mask = torch.triu(

                torch.ones(

                    context_length,

                    context_length,

                    dtype=torch.bool
                ),

                diagonal=1
            )


            self.register_buffer(

                "causal_mask",

                causal_mask,

                persistent=False
            )


    def forward(
        self,
        x
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
                head_size
            )

            .transpose(
                1,
                2
            )
        )


        K = (

            K.reshape(

                B,
                T,
                number_of_heads,
                head_size
            )

            .transpose(
                1,
                2
            )
        )


        V = (

            V.reshape(

                B,
                T,
                number_of_heads,
                head_size
            )

            .transpose(
                1,
                2
            )
        )


        Q, K = self.rotary(
            Q,
            K
        )


        # ====================================================
        # OLD EDUCATIONAL ATTENTION
        # ====================================================

        if (
            self.attention_mode
            ==
            "manual"
        ):

            scores = (

                Q

                @

                K.transpose(
                    -2,
                    -1
                )
            )


            scores = (

                scores

                /

                math.sqrt(
                    head_size
                )
            )


            scores = scores.masked_fill(

                self.causal_mask[
                    :T,
                    :T
                ],

                float("-inf")
            )


            attention = F.softmax(

                scores,

                dim=-1
            )


            out = (

                attention

                @

                V
            )


        # ====================================================
        # OPTIMIZED PYTORCH SDPA
        # ====================================================

        elif (
            self.attention_mode
            ==
            "sdpa"
        ):

            out = F.scaled_dot_product_attention(

                Q,

                K,

                V,

                attn_mask=None,

                dropout_p=0.0,

                is_causal=True
            )


        else:

            raise ValueError(
                self.attention_mode
            )


        out = (

            out

            .transpose(
                1,
                2
            )

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
# 12. SWIGLU
# ============================================================

class SwiGLU(
    nn.Module
):

    def __init__(self):

        super().__init__()


        self.gate = nn.Linear(

            d_model,

            mlp_hidden,

            bias=False
        )


        self.up = nn.Linear(

            d_model,

            mlp_hidden,

            bias=False
        )


        self.down = nn.Linear(

            mlp_hidden,

            d_model,

            bias=False
        )


    def forward(
        self,
        x
    ):

        return self.down(

            F.silu(
                self.gate(x)
            )

            *

            self.up(x)
        )


# ============================================================
# 13. TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(
    nn.Module
):

    def __init__(
        self,
        attention_mode
    ):

        super().__init__()


        self.attention_norm = RMSNorm(
            d_model
        )


        self.attention = (
            CausalSelfAttention(
                attention_mode
            )
        )


        self.mlp_norm = RMSNorm(
            d_model
        )


        self.mlp = SwiGLU()


    def forward(
        self,
        x
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
# 14. BLANK-0
# ============================================================

class BlankZero(
    nn.Module
):

    def __init__(
        self,
        attention_mode
    ):

        super().__init__()


        self.token_embedding = nn.Embedding(

            vocab_size,

            d_model
        )


        self.blocks = nn.ModuleList([

            TransformerBlock(
                attention_mode
            )

            for _ in range(
                number_of_layers
            )

        ])


        self.final_norm = RMSNorm(
            d_model
        )


    def forward(
        self,
        X,
        targets=None
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


        logits = F.linear(

            x,

            self.token_embedding.weight
        )


        loss = None


        if targets is not None:

            loss = F.cross_entropy(

                logits.reshape(
                    -1,
                    vocab_size
                ),

                targets.reshape(
                    -1
                )
            )


        return logits, loss


# ============================================================
# 15. LOAD TRAINED v0.2 WEIGHTS ON CPU
# ============================================================

print(
    "\nLoading Blank-0 v0.2 best checkpoint..."
)


checkpoint = torch.load(

    CHECKPOINT_FILE,

    map_location="cpu",

    weights_only=False
)


checkpoint_state = checkpoint[
    "model_state_dict"
]


print(
    "Checkpoint update:",
    checkpoint[
        "completed_updates"
    ]
)


print(
    "Checkpoint validation loss:",
    checkpoint[
        "best_validation_loss"
    ]
)


# ============================================================
# 16. AUTOCAST HELPER
# ============================================================

def autocast_context(
    precision
):

    if precision == "fp32":

        return torch.autocast(

            device_type="cuda",

            enabled=False
        )


    elif precision == "bf16":

        return torch.autocast(

            device_type="cuda",

            dtype=torch.bfloat16
        )


    elif precision == "fp16":

        return torch.autocast(

            device_type="cuda",

            dtype=torch.float16
        )


    else:

        raise ValueError(
            precision
        )


# ============================================================
# 17. FIXED-BATCH LOSS TEST
# ============================================================

fixed_X, fixed_Y = make_batch(

    all_start_positions[
        0
    ]
)


reference_loss = None


# ============================================================
# 18. BENCHMARK FUNCTION
# ============================================================

def benchmark_variant(
    name,
    attention_mode,
    precision
):

    global reference_loss


    print(
        "\n"
        +
        "=" * 70
    )

    print(
        name
    )

    print(
        "=" * 70
    )


    gc.collect()

    torch.cuda.empty_cache()


    # --------------------------------------------------------
    # Fresh model
    # --------------------------------------------------------

    model = BlankZero(

        attention_mode
    ).to(
        device
    )


    model.load_state_dict(
        checkpoint_state
    )


    model.train()


    parameter_count = sum(

        parameter.numel()

        for parameter in model.parameters()
    )


    print(
        "Parameters:",
        f"{parameter_count:,}"
    )


    # --------------------------------------------------------
    # Numerical sanity test BEFORE optimizer updates
    # --------------------------------------------------------

    model.eval()


    with torch.no_grad():

        with autocast_context(
            precision
        ):

            _, sanity_loss = model(

                fixed_X,

                fixed_Y
            )


    sanity_loss_value = (
        sanity_loss.item()
    )


    print(
        "Fixed-batch loss:",
        f"{sanity_loss_value:.6f}"
    )


    if reference_loss is None:

        reference_loss = (
            sanity_loss_value
        )


        print(
            "Loss difference from baseline:",
            "BASELINE"
        )

    else:

        difference = (

            sanity_loss_value

            -

            reference_loss
        )


        print(
            "Loss difference from baseline:",
            f"{difference:+.6f}"
        )


    model.train()


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=1e-4,

        betas=(
            0.9,
            0.95
        ),

        weight_decay=0.1
    )


    # --------------------------------------------------------
    # FP16 needs dynamic loss scaling.
    #
    # BF16 generally does not.
    # --------------------------------------------------------

    scaler = None


    if precision == "fp16":

        try:

            scaler = torch.amp.GradScaler(

                "cuda"
            )

        except Exception:

            scaler = torch.cuda.amp.GradScaler()


    # --------------------------------------------------------
    # Training step
    # --------------------------------------------------------

    def run_update(
        starts
    ):

        X, Y = make_batch(
            starts
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with autocast_context(
            precision
        ):

            _, loss = model(
                X,
                Y
            )


        if scaler is not None:

            scaler.scale(
                loss
            ).backward()


            scaler.unscale_(
                optimizer
            )


            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                1.0
            )


            scaler.step(
                optimizer
            )


            scaler.update()


        else:

            loss.backward()


            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                1.0
            )


            optimizer.step()


        return loss


    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print(
        "Warmup..."
    )


    for index in range(
        WARMUP_UPDATES
    ):

        run_update(

            all_start_positions[
                index + 1
            ]
        )


    torch.cuda.synchronize()


    # --------------------------------------------------------
    # Reset peak memory AFTER warmup
    # --------------------------------------------------------

    torch.cuda.reset_peak_memory_stats(
        0
    )


    # --------------------------------------------------------
    # Timed section
    # --------------------------------------------------------

    print(
        "Benchmarking..."
    )


    start_time = (
        time.perf_counter()
    )


    last_loss = None


    for index in range(
        TIMED_UPDATES
    ):

        batch_index = (

            WARMUP_UPDATES

            +

            index

            +

            1
        )


        last_loss = run_update(

            all_start_positions[
                batch_index
            ]
        )


    torch.cuda.synchronize()


    elapsed = (

        time.perf_counter()

        -

        start_time
    )


    total_timed_tokens = (

        TIMED_UPDATES

        *

        TOKENS_PER_UPDATE
    )


    tokens_per_second = (

        total_timed_tokens

        /

        elapsed
    )


    milliseconds_per_update = (

        elapsed

        /
        TIMED_UPDATES

        *
        1000
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


    print(
        "\nRESULT"
    )


    print(
        "Precision:",
        precision
    )


    print(
        "Attention:",
        attention_mode
    )


    print(
        "Elapsed:",
        round(
            elapsed,
            3
        ),
        "sec"
    )


    print(
        "ms/update:",
        round(
            milliseconds_per_update,
            2
        )
    )


    print(
        "Tokens/sec:",
        f"{tokens_per_second:,.0f}"
    )


    print(
        "Last training loss:",
        round(
            last_loss.item(),
            6
        )
    )


    print(
        "Peak allocated VRAM:",
        round(
            peak_allocated,
            2
        ),
        "GB"
    )


    print(
        "Peak reserved VRAM:",
        round(
            peak_reserved,
            2
        ),
        "GB"
    )


    result = {

        "name":
            name,

        "attention":
            attention_mode,

        "precision":
            precision,

        "sanity_loss":
            sanity_loss_value,

        "loss_difference":
            (
                sanity_loss_value
                -
                reference_loss
            ),

        "tokens_per_second":
            tokens_per_second,

        "milliseconds_per_update":
            milliseconds_per_update,

        "peak_allocated_gb":
            peak_allocated,

        "peak_reserved_gb":
            peak_reserved
    }


    del optimizer
    del model


    if scaler is not None:

        del scaler


    gc.collect()

    torch.cuda.empty_cache()

    torch.cuda.synchronize()


    return result


# ============================================================
# 19. RUN BENCHMARKS
# ============================================================

variants = [

    (
        "1. Manual attention + FP32",
        "manual",
        "fp32"
    ),

    (
        "2. SDPA attention + FP32",
        "sdpa",
        "fp32"
    ),

    (
        "3. SDPA attention + BF16 autocast",
        "sdpa",
        "bf16"
    ),

    (
        "4. SDPA attention + FP16 autocast",
        "sdpa",
        "fp16"
    )
]


results = []


for (
    name,
    attention_mode,
    precision
) in variants:

    try:

        result = benchmark_variant(

            name,

            attention_mode,

            precision
        )


        results.append(
            result
        )


    except Exception as error:

        print(
            "\nFAILED:"
        )

        print(
            name
        )

        print(
            type(error).__name__,
            ":",
            error
        )


        gc.collect()

        torch.cuda.empty_cache()


# ============================================================
# 20. FINAL COMPARISON
# ============================================================

print(
    "\n"
    +
    "=" * 90
)

print(
    "FINAL PERFORMANCE COMPARISON"
)

print(
    "=" * 90
)


if len(results) == 0:

    raise RuntimeError(
        "Every benchmark failed."
    )


baseline_speed = results[
    0
][
    "tokens_per_second"
]


print()

print(

    f"{'Variant':38s} "
    f"{'tok/s':>12s} "
    f"{'speedup':>10s} "
    f"{'loss diff':>12s} "
    f"{'peak GB':>10s}"
)


print(
    "-" * 88
)


for result in results:

    speedup = (

        result[
            "tokens_per_second"
        ]

        /

        baseline_speed
    )


    print(

        f"{result['name']:38s} "

        f"{result['tokens_per_second']:>12,.0f} "

        f"{speedup:>9.2f}x "

        f"{result['loss_difference']:>+12.6f} "

        f"{result['peak_reserved_gb']:>10.2f}"
    )


# ============================================================
# 21. FASTEST RESULT
# ============================================================

fastest = max(

    results,

    key=lambda result:
        result[
            "tokens_per_second"
        ]
)


print(
    "\nFASTEST:"
)

print(
    fastest[
        "name"
    ]
)


print(
    "Speed:",
    f"{fastest['tokens_per_second']:,.0f}",
    "tokens/sec"
)


print(
    "Speedup:",
    round(

        fastest[
            "tokens_per_second"
        ]

        /

        baseline_speed,

        2
    ),
    "x"
)


print(
    "Loss difference:",
    f"{fastest['loss_difference']:+.6f}"
)


print(
    "\nDONE"
)