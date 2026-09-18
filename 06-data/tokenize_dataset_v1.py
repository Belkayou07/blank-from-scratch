import json
import os
import time

import numpy as np

from tokenizers import Tokenizer


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = "blank_dataset_v1.txt"

TOKENIZER_FILE = "blank_tokenizer_v1.json"

OUTPUT_FILE = "blank_dataset_v1_tokens.bin"

METADATA_FILE = "blank_dataset_v1_tokens_meta.json"


# Process a few million characters at a time.
#
# This prevents us from creating one gigantic
# Python list containing ~30 million integers.

CHUNK_CHARACTERS = 4_000_000


# ============================================================
# 2. CHECK FILES
# ============================================================

if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        DATA_FILE
    )


if not os.path.exists(TOKENIZER_FILE):

    raise FileNotFoundError(
        TOKENIZER_FILE
    )


# ============================================================
# 3. LOAD TOKENIZER
# ============================================================

tokenizer = Tokenizer.from_file(
    TOKENIZER_FILE
)


vocab_size = (
    tokenizer.get_vocab_size()
)


print(
    "Tokenizer vocabulary:",
    f"{vocab_size:,}"
)


# ============================================================
# 4. CHOOSE STORAGE TYPE
# ============================================================
#
# uint16 supports:
#
# 0 ... 65,535
#
# Our vocabulary contains only 4,096 tokens,
# so uint16 is more than enough.

if vocab_size <= 65_536:

    token_dtype = np.uint16

    dtype_name = "uint16"

else:

    token_dtype = np.uint32

    dtype_name = "uint32"


print(
    "Token storage type:",
    dtype_name
)


# ============================================================
# 5. SOURCE INFORMATION
# ============================================================

source_bytes = os.path.getsize(
    DATA_FILE
)


print(
    "Source dataset:",
    DATA_FILE
)


print(
    "Source size:",
    round(
        source_bytes / 1024**2,
        2
    ),
    "MB"
)


# ============================================================
# 6. TOKENIZE DATASET
# ============================================================
#
# IMPORTANT:
#
# We use newline=""
#
# because Windows otherwise translates
#
# \r\n
#
# into
#
# \n
#
# while reading.
#
# Our tokenizer was trained on the actual file,
# including its Windows CRLF sequences, so we
# preserve those here.

print(
    "\nTOKENIZING DATASET\n"
)


start_time = time.perf_counter()


total_tokens = 0

total_characters = 0

chunk_number = 0


with open(
    DATA_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as input_file:

    with open(
        OUTPUT_FILE,
        "wb"
    ) as output_file:

        while True:

            text = input_file.read(
                CHUNK_CHARACTERS
            )


            if not text:
                break


            chunk_number += 1


            # ====================================
            # TOKENIZE
            # ====================================

            encoding = tokenizer.encode(
                text
            )


            token_ids = (
                encoding.ids
            )


            # ====================================
            # CONVERT TO COMPACT INTEGER ARRAY
            # ====================================

            token_array = np.asarray(
                token_ids,
                dtype=token_dtype
            )


            # ====================================
            # WRITE RAW BINARY TOKEN IDs
            # ====================================

            output_file.write(
                token_array.tobytes()
            )


            total_tokens += len(
                token_ids
            )


            total_characters += len(
                text
            )


            elapsed = (
                time.perf_counter()
                -
                start_time
            )


            chars_per_token = (
                total_characters
                /
                total_tokens
            )


            print(
                f"chunk={chunk_number:3d} | "
                f"chars={total_characters:>11,} | "
                f"tokens={total_tokens:>10,} | "
                f"chars/token={chars_per_token:.3f} | "
                f"time={elapsed:.1f}s"
            )


# ============================================================
# 7. FINAL STATISTICS
# ============================================================

end_time = time.perf_counter()


elapsed = (
    end_time
    -
    start_time
)


binary_size = os.path.getsize(
    OUTPUT_FILE
)


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


print(
    "\nTOKENIZATION COMPLETE"
)


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
    round(
        characters_per_token,
        4
    )
)


print(
    "Tokens/character:",
    round(
        tokens_per_character,
        4
    )
)


print(
    "Binary dataset size:",
    round(
        binary_size / 1024**2,
        2
    ),
    "MB"
)


print(
    "Tokenization time:",
    round(
        elapsed,
        2
    ),
    "seconds"
)


print(
    "Tokens/second:",
    f"{total_tokens / elapsed:,.0f}"
)


# ============================================================
# 8. TRAIN / VALIDATION SPLIT INFORMATION
# ============================================================
#
# We don't duplicate the binary file.
#
# The training program will memory-map this file and
# logically divide it into:
#
# first 98% -> training
# final 2%  -> validation

train_fraction = 0.98


train_tokens = int(
    total_tokens
    *
    train_fraction
)


validation_tokens = (
    total_tokens
    -
    train_tokens
)


print(
    "\nDATA SPLIT"
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
    "Training percentage:",
    "98%"
)


print(
    "Validation percentage:",
    "2%"
)


# ============================================================
# 9. SAVE METADATA
# ============================================================

metadata = {

    "dataset_file":
        OUTPUT_FILE,

    "tokenizer_file":
        TOKENIZER_FILE,

    "vocab_size":
        vocab_size,

    "dtype":
        dtype_name,

    "total_characters":
        total_characters,

    "total_tokens":
        total_tokens,

    "characters_per_token":
        characters_per_token,

    "train_fraction":
        train_fraction,

    "train_tokens":
        train_tokens,

    "validation_tokens":
        validation_tokens
}


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as metadata_file:

    json.dump(
        metadata,
        metadata_file,
        indent=2
    )


print(
    "\nMetadata saved:",
    METADATA_FILE
)


# ============================================================
# 10. VERIFY BINARY DATASET
# ============================================================

print(
    "\nVERIFYING BINARY DATASET"
)


# Memory-map without loading the entire dataset
# into RAM.

tokens = np.memmap(

    OUTPUT_FILE,

    dtype=token_dtype,

    mode="r"
)


print(
    "Tokens visible through memmap:",
    f"{len(tokens):,}"
)


print(
    "Expected tokens:",
    f"{total_tokens:,}"
)


if len(tokens) != total_tokens:

    raise RuntimeError(
        "Binary token count does not match!"
    )


# ============================================================
# 11. SAMPLE DECODE
# ============================================================

sample_size = min(
    100,
    len(tokens)
)


sample_ids = (

    tokens[
        :sample_size
    ]

    .astype(
        np.int64
    )

    .tolist()
)


sample_text = tokenizer.decode(
    sample_ids
)


print(
    "\nFIRST 100 TOKENS DECODE TO:\n"
)


print(
    sample_text
)


print(
    "\nBinary dataset verification: OK"
)


# ============================================================
# 12. CONTEXT SCALE
# ============================================================

print(
    "\nAPPROXIMATE TEXT PER CONTEXT"
)


for context_length in [
    256,
    512,
    1024,
    2048
]:

    estimated_characters = (
        context_length
        *
        characters_per_token
    )


    print(
        f"{context_length:4d} tokens "
        f"≈ "
        f"{estimated_characters:,.0f} characters"
    )


print(
    "\nDONE"
)