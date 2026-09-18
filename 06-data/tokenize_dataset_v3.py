import json
import os
import time

import numpy as np

from tokenizers import Tokenizer


# ============================================================
# 1. FILES
# ============================================================

TEXT_FILE = "blank_dataset_v3.txt"

TOKENIZER_FILE = "blank_tokenizer_v1.json"

OUTPUT_FILE = "blank_dataset_v3_tokens.bin"

METADATA_FILE = "blank_dataset_v3_tokens_meta.json"


# ============================================================
# 2. CONFIG
# ============================================================

# Large enough for good throughput without making memory use
# excessive.

CHUNK_CHARACTERS = 8_000_000


TRAIN_FRACTION = 0.98


# ============================================================
# 3. VERIFY INPUTS
# ============================================================

for filename in [
    TEXT_FILE,
    TOKENIZER_FILE,
]:

    if not os.path.exists(filename):

        raise FileNotFoundError(
            filename
        )


if os.path.exists(OUTPUT_FILE):

    raise FileExistsError(
        f"{OUTPUT_FILE} already exists.\n"
        "Delete or rename it before tokenizing Dataset v3."
    )


# ============================================================
# 4. LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")


tokenizer = Tokenizer.from_file(
    TOKENIZER_FILE
)


vocab_size = tokenizer.get_vocab_size()


print(
    "Vocabulary:",
    f"{vocab_size:,}"
)


if vocab_size > 65_535:

    raise RuntimeError(
        "Vocabulary is too large for uint16."
    )


# ============================================================
# 5. SOURCE SIZE
# ============================================================

source_size_bytes = os.path.getsize(
    TEXT_FILE
)


print()
print("BLANK DATASET v3 TOKENIZATION")
print("=============================")

print(
    "Source file:",
    TEXT_FILE
)

print(
    "Source size:",
    f"{source_size_bytes / 1024**2:,.2f} MB"
)

print(
    "Source size:",
    f"{source_size_bytes / 1024**3:.3f} GB"
)

print(
    "Tokenizer:",
    TOKENIZER_FILE
)

print(
    "Output dtype: uint16"
)

print(
    "Chunk characters:",
    f"{CHUNK_CHARACTERS:,}"
)

print()


# ============================================================
# 6. TOKENIZE
# ============================================================
#
# Important:
#
# We open with newline="" so Python does not perform newline
# translation.
#
# Each 8M-character chunk is encoded independently.
#
# This means BPE merges cannot cross a chunk boundary, but
# there will only be roughly ~375 boundaries across 3B chars.
#
# That effect is completely negligible for this project.
# ============================================================

start_time = time.perf_counter()


total_characters = 0

total_tokens = 0

chunks_processed = 0


progress_interval = (
    100_000_000
)

next_progress = (
    progress_interval
)


with open(
    TEXT_FILE,
    "r",
    encoding="utf-8",
    newline="",
) as source:

    with open(
        OUTPUT_FILE,
        "wb",
    ) as output:

        while True:

            text = source.read(
                CHUNK_CHARACTERS
            )


            if not text:

                break


            character_count = len(
                text
            )


            # ------------------------------------------------
            # Encode
            # ------------------------------------------------

            encoding = tokenizer.encode(
                text
            )


            token_ids = encoding.ids


            # ------------------------------------------------
            # Convert directly to uint16
            # ------------------------------------------------

            token_array = np.asarray(

                token_ids,

                dtype=np.uint16,
            )


            # ------------------------------------------------
            # Write binary tokens
            # ------------------------------------------------

            token_array.tofile(
                output
            )


            total_characters += (
                character_count
            )


            total_tokens += (
                len(token_array)
            )


            chunks_processed += 1


            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                total_characters
                >=
                next_progress
            ):

                elapsed = (

                    time.perf_counter()

                    -

                    start_time
                )


                chars_per_second = (

                    total_characters

                    /

                    elapsed
                )


                tokens_per_second = (

                    total_tokens

                    /

                    elapsed
                )


                percent = (

                    total_characters

                    /

                    3_000_000_000

                    *

                    100
                )


                print(

                    f"{total_characters:,} chars | "

                    f"{total_tokens:,} tokens | "

                    f"{percent:.1f}% | "

                    f"{chars_per_second / 1_000_000:.2f} M chars/s | "

                    f"{tokens_per_second:,.0f} tok/s"
                )


                while (
                    next_progress
                    <=
                    total_characters
                ):

                    next_progress += (
                        progress_interval
                    )


# ============================================================
# 7. TIMING
# ============================================================

elapsed = (

    time.perf_counter()

    -

    start_time
)


# ============================================================
# 8. BASIC METRICS
# ============================================================

characters_per_token = (

    total_characters

    /

    total_tokens
)


tokens_per_character = (

    total_tokens

    /

    total_characters
)


binary_size_bytes = os.path.getsize(
    OUTPUT_FILE
)


binary_size_mb = (

    binary_size_bytes

    /

    1024**2
)


binary_size_gb = (

    binary_size_bytes

    /

    1024**3
)


expected_binary_size = (

    total_tokens

    *

    np.dtype(
        np.uint16
    ).itemsize
)


if binary_size_bytes != expected_binary_size:

    raise RuntimeError(

        f"Binary size mismatch.\n"

        f"Expected: "
        f"{expected_binary_size:,} bytes\n"

        f"Actual: "
        f"{binary_size_bytes:,} bytes"
    )


# ============================================================
# 9. TRAIN / VALIDATION SPLIT
# ============================================================

train_tokens = int(

    total_tokens

    *

    TRAIN_FRACTION
)


validation_tokens = (

    total_tokens

    -

    train_tokens
)


# ============================================================
# 10. MEMORY-MAP VERIFICATION
# ============================================================

print()
print("Verifying binary token file...")


mapped = np.memmap(

    OUTPUT_FILE,

    dtype=np.uint16,

    mode="r",
)


if len(mapped) != total_tokens:

    raise RuntimeError(

        f"Memmap token count mismatch.\n"

        f"Expected: {total_tokens:,}\n"

        f"Found: {len(mapped):,}"
    )


minimum_token = int(
    mapped.min()
)


maximum_token = int(
    mapped.max()
)


if minimum_token < 0:

    raise RuntimeError(
        "Negative token ID found."
    )


if maximum_token >= vocab_size:

    raise RuntimeError(

        f"Token ID {maximum_token} exceeds "
        f"vocabulary size {vocab_size}."
    )


# ============================================================
# 11. DECODE SAMPLE
# ============================================================

sample_count = min(

    150,

    total_tokens
)


sample_ids = (

    mapped[
        :sample_count
    ]

    .astype(
        np.int64
    )

    .tolist()
)


sample_text = tokenizer.decode(
    sample_ids
)


# ============================================================
# 12. EFFECTIVE CONTEXT ESTIMATES
# ============================================================

contexts = {}


for context in [

    256,
    512,
    1024,
    2048,
    4096,

]:

    contexts[
        str(context)
    ] = (

        context

        *

        characters_per_token
    )


# ============================================================
# 13. METADATA
# ============================================================

metadata = {

    "dataset_name":
        "Blank Dataset v3 Tokens",

    "source_text_file":
        TEXT_FILE,

    "tokenizer_file":
        TOKENIZER_FILE,

    "dtype":
        "uint16",

    "vocab_size":
        vocab_size,

    "total_characters":
        total_characters,

    "total_tokens":
        total_tokens,

    "characters_per_token":
        characters_per_token,

    "tokens_per_character":
        tokens_per_character,

    "train_fraction":
        TRAIN_FRACTION,

    "train_tokens":
        train_tokens,

    "validation_tokens":
        validation_tokens,

    "chunk_characters":
        CHUNK_CHARACTERS,

    "chunks_processed":
        chunks_processed,

    "binary_size_bytes":
        binary_size_bytes,

    "minimum_token_id":
        minimum_token,

    "maximum_token_id":
        maximum_token,

    "tokenization_seconds":
        elapsed,

    "tokens_per_second":
        total_tokens / elapsed,

    "characters_per_second":
        total_characters / elapsed,

    "effective_context_characters":
        contexts,
}


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(

        metadata,

        f,

        indent=2,
    )


# ============================================================
# 14. REPORT
# ============================================================

print()
print("=" * 72)
print("DATASET v3 TOKENIZATION COMPLETE")
print("=" * 72)


print(
    "Characters:",
    f"{total_characters:,}"
)


print(
    "Tokens:",
    f"{total_tokens:,}"
)


print(
    "Characters/token:",
    f"{characters_per_token:.4f}"
)


print(
    "Tokens/character:",
    f"{tokens_per_character:.4f}"
)


print(
    "Binary size:",
    f"{binary_size_mb:,.2f} MB"
)


print(
    "Binary size:",
    f"{binary_size_gb:.3f} GB"
)


print(
    "Chunks:",
    f"{chunks_processed:,}"
)


print()
print(
    "Training tokens:",
    f"{train_tokens:,}"
)


print(
    "Validation tokens:",
    f"{validation_tokens:,}"
)


print()
print(
    "Tokenization time:",
    f"{elapsed:.2f} seconds"
)


print(
    "Throughput:",
    f"{total_tokens / elapsed:,.0f} tokens/sec"
)


print()
print(
    "Token range:",
    f"{minimum_token} -> {maximum_token}"
)


print()
print("Effective context size:")


for context in [

    256,
    512,
    1024,
    2048,
    4096,

]:

    print(

        f"{context:4d} tokens "
        f"≈ "
        f"{contexts[str(context)]:,.0f} characters"
    )


print()
print("First 150-token decode:")
print("-" * 72)

print(
    sample_text
)

print("-" * 72)


print()
print(
    "Token file:",
    OUTPUT_FILE
)


print(
    "Metadata:",
    METADATA_FILE
)


print()
print(
    "Binary-size verification: PASSED"
)

print(
    "Memmap verification: PASSED"
)

print(
    "Token-range verification: PASSED"
)

print(
    "DONE"
)