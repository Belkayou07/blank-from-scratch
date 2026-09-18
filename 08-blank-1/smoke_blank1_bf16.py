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


for filename in [
    DATA_FILE,
    METADATA_FILE
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
# 3. LOAD METADATA
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
# 4. BLANK-1 ARCHITECTURE
# ============================================================

context_length = 512

batch_size = 16


d_model = 512

number_of_layers = 10

number_of_heads = 8

head_size = (
    d_model
    //
    number_of_heads
)


mlp_hidden = 1365


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


tokens_per_update = (
    batch_size
    *
    context_length
)


# ============================================================
# 5. SMOKE-TEST TRAINING CONFIG
# ============================================================

training_updates = 500

warmup_updates = 100

maximum_learning_rate = 2.5e-4

weight_decay = 0.1

gradient_clip = 1.0


evaluation_interval = 100

evaluation_batches = 10


print(
    "\nBLANK-1 BF16 SMOKE TEST"
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
    "Smoke-test updates:",
    training_updates
)

print(
    "Tokens/update:",
    f"{tokens_per_update:,}"
)

print(
    "Total token predictions:",
    f"{training_updates * tokens_per_update:,}"
)


# ============================================================
# 6. RANDOM SEEDS
# ============================================================

torch.manual_seed(42)

torch.cuda.manual_seed_all(42)


# ============================================================
# 7. LOAD DATASET
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
    "Converting token IDs to int64..."
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


validation_data = all_tokens[
    train_tokens:
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
# 8. RANDOM BATCHES
# ============================================================

batch_offsets = torch.arange(

    context_length,

    device=device
)


def get_batch(
    split
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

        device=device
    )


    positions = (

        starts[:, None]

        +

        batch_offsets[
            None,
            :
        ]
    )


    X = source[
        positions
    ]


    Y = source[
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
# 11. SDPA SELF-ATTENTION
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


        self.rotary = RotaryEmbedding(

            head_size,

            context_length
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


        out = F.scaled_dot_product_attention(

            Q,

            K,

            V,

            attn_mask=None,

            dropout_p=0.0,

            is_causal=True
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
# 12. SwiGLU
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

    def __init__(self):

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
# 14. BLANK-1
# ============================================================

class BlankOne(
    nn.Module
):

    def __init__(self):

        super().__init__()


        self.token_embedding = nn.Embedding(

            vocab_size,

            d_model
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
        module
    ):

        if isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(

                module.weight,

                mean=0.0,

                std=0.02
            )


        elif isinstance(
            module,
            nn.Embedding
        ):

            nn.init.normal_(

                module.weight,

                mean=0.0,

                std=0.02
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
# 15. CREATE MODEL
# ============================================================

model = BlankOne().to(
    device
)


parameter_count = sum(

    parameter.numel()

    for parameter in model.parameters()
)


print(
    "\nTRAINABLE PARAMETERS:",
    f"{parameter_count:,}"
)


EXPECTED_PARAMETERS = (
    33_560_064
)


if parameter_count != EXPECTED_PARAMETERS:

    raise RuntimeError(

        f"Expected "
        f"{EXPECTED_PARAMETERS:,} "
        f"parameters, got "
        f"{parameter_count:,}."
    )


print(
    "Parameter-count check: OK"
)


print(
    "Blank-0 parameters: 15,735,168"
)

print(
    "Blank-1 / Blank-0:",
    round(
        parameter_count
        /
        15_735_168,
        3
    ),
    "x"
)


# ============================================================
# 16. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=maximum_learning_rate,

    betas=(
        0.9,
        0.95
    ),

    weight_decay=weight_decay
)


# ============================================================
# 17. LEARNING RATE
# ============================================================

def get_learning_rate(
    update_index
):

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


    return maximum_learning_rate


# ============================================================
# 18. BF16 CONTEXT
# ============================================================

def bf16_context():

    return torch.autocast(

        device_type="cuda",

        dtype=torch.bfloat16
    )


# ============================================================
# 19. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate():

    model.eval()


    results = {}


    for split in [
        "train",
        "validation"
    ]:

        losses = []


        for _ in range(
            evaluation_batches
        ):

            X, Y = get_batch(
                split
            )


            with bf16_context():

                _, loss = model(
                    X,
                    Y
                )


            losses.append(
                loss.item()
            )


        average_loss = (

            sum(losses)

            /

            len(losses)
        )


        results[
            split
        ] = average_loss


    model.train()


    return results


# ============================================================
# 20. INITIAL EVALUATION
# ============================================================

print(
    "\nINITIAL EVALUATION"
)


initial_metrics = evaluate()


expected_random_loss = math.log(
    vocab_size
)


print(
    "Expected random CE ≈ ln(vocab):",
    round(
        expected_random_loss,
        4
    )
)


print(
    "Train loss:",
    round(
        initial_metrics[
            "train"
        ],
        4
    )
)


print(
    "Validation loss:",
    round(
        initial_metrics[
            "validation"
        ],
        4
    )
)


if not math.isfinite(
    initial_metrics[
        "validation"
    ]
):

    raise RuntimeError(
        "Initial validation loss is not finite."
    )


# ============================================================
# 21. TRAINING
# ============================================================

print(
    "\nTRAINING SMOKE TEST\n"
)


model.train()


torch.cuda.reset_peak_memory_stats(
    0
)


session_start = (
    time.perf_counter()
)


segment_start = (
    time.perf_counter()
)


segment_updates = 0


for update in range(
    training_updates
):

    learning_rate = get_learning_rate(
        update
    )


    for group in optimizer.param_groups:

        group[
            "lr"
        ] = learning_rate


    X, Y = get_batch(
        "train"
    )


    optimizer.zero_grad(
        set_to_none=True
    )


    with bf16_context():

        _, loss = model(
            X,
            Y
        )


    if not torch.isfinite(
        loss
    ):

        raise RuntimeError(

            f"Non-finite loss at "
            f"update {update + 1}: "
            f"{loss.item()}"
        )


    loss.backward()


    gradient_norm = (
        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            gradient_clip
        )
    )


    if not torch.isfinite(
        gradient_norm
    ):

        raise RuntimeError(

            f"Non-finite gradient norm "
            f"at update {update + 1}."
        )


    optimizer.step()


    segment_updates += 1


    completed_updates = (
        update
        +
        1
    )


    if (
        completed_updates
        %
        evaluation_interval
        == 0
    ):

        torch.cuda.synchronize()


        now = time.perf_counter()


        elapsed = (

            now

            -

            segment_start
        )


        segment_tokens = (

            segment_updates

            *

            tokens_per_update
        )


        throughput = (

            segment_tokens

            /

            elapsed
        )


        metrics = evaluate()


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

            f"updates={completed_updates:4d} | "

            f"train={metrics['train']:.4f} | "

            f"val={metrics['validation']:.4f} | "

            f"batch_loss={loss.item():.4f} | "

            f"grad={gradient_norm.item():.3f} | "

            f"lr={learning_rate:.2e} | "

            f"tok/s={throughput:,.0f} | "

            f"VRAM={peak_reserved:.2f} GB"
        )


        if not math.isfinite(
            metrics[
                "validation"
            ]
        ):

            raise RuntimeError(
                "Validation loss became non-finite."
            )


        segment_start = (
            time.perf_counter()
        )


        segment_updates = 0


# ============================================================
# 22. FINAL RESULTS
# ============================================================

torch.cuda.synchronize()


session_elapsed = (

    time.perf_counter()

    -

    session_start
)


final_metrics = evaluate()


total_predictions = (

    training_updates

    *

    tokens_per_update
)


average_training_throughput = (

    total_predictions

    /

    session_elapsed
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
    "\n"
    +
    "=" * 70
)

print(
    "SMOKE TEST COMPLETE"
)

print(
    "=" * 70
)


print(
    "Parameters:",
    f"{parameter_count:,}"
)


print(
    "Completed updates:",
    training_updates
)


print(
    "Token predictions:",
    f"{total_predictions:,}"
)


print(
    "Initial train loss:",
    round(
        initial_metrics[
            "train"
        ],
        4
    )
)


print(
    "Initial validation loss:",
    round(
        initial_metrics[
            "validation"
        ],
        4
    )
)


print(
    "Final train loss:",
    round(
        final_metrics[
            "train"
        ],
        4
    )
)


print(
    "Final validation loss:",
    round(
        final_metrics[
            "validation"
        ],
        4
    )
)


print(
    "Total session time:",
    round(
        session_elapsed,
        2
    ),
    "seconds"
)


print(
    "Whole-session average:",
    f"{average_training_throughput:,.0f}",
    "tokens/sec"
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


# ============================================================
# 23. AUTOMATIC HEALTH CHECK
# ============================================================

loss_improved = (

    final_metrics[
        "validation"
    ]

    <

    initial_metrics[
        "validation"
    ]
)


finite_final_loss = math.isfinite(

    final_metrics[
        "validation"
    ]
)


print(
    "\nHEALTH CHECK"
)


print(
    "Finite final loss:",
    finite_final_loss
)


print(
    "Validation loss decreased:",
    loss_improved
)


if (
    finite_final_loss
    and
    loss_improved
):

    print(
        "\nRESULT: BLANK-1 BF16 TRAINING LOOKS HEALTHY."
    )

else:

    print(
        "\nRESULT: SOMETHING NEEDS INVESTIGATION."
    )


print(
    "\nNo checkpoint was saved."
)

print(
    "This was only the Blank-1 smoke test."
)

print(
    "\nDONE"
)