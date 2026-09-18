import json
import os
import time

import numpy as np

from tokenizers import Tokenizer


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = "blank_dataset_v2.txt"

TOKENIZER_FILE = "blank_tokenizer_v1.json"

OUTPUT_FILE = "blank_dataset_v2_tokens.bin"

METADATA_FILE = "blank_dataset_v2_tokens_meta.json"


# Larger chunks than v1.
#
# 8M characters is still very manageable on your PC,
# while reducing the number of artificial chunk boundaries.

CHUNK_CHARACTERS = 8_000_000


# Keep the same logical split strategy.

TRAIN_FRACTION = 0.98


# ============================================================
# 2. CHECK REQUIRED FILES
# ============================================================

for filename in [
    DATA_FILE,
    TOKENIZER_FILE
]:

    if not os.path.exists(filename):

        raise FileNotFoundError(
            filename
        )


# ============================================================
# 3. LOAD OUR EXISTING TOKENIZER
# ============================================================

print(
    "Loading tokenizer..."
)


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
# 4. STORAGE DATA TYPE
# ============================================================
#
# Vocabulary:
#
# 0 ... 4095
#
# uint16 supports:
#
# 0 ... 65535
#
# Therefore every token still requires only 2 bytes.

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
# 5. SOURCE DATASET INFO
# ============================================================

source_size = os.path.getsize(
    DATA_FILE
)


print(
    "Source dataset:",
    DATA_FILE
)


print(
    "Source size:",
    round(
        source_size / 1024**2,
        2
    ),
    "MB"
)


print(
    "Source size:",
    round(
        source_size / 1024**3,
        3
    ),
    "GB"
)


# ============================================================
# 6. TOKENIZE
# ============================================================

print(
    "\nTOKENIZING DATASET V2\n"
)


start_time = (
    time.perf_counter()
)


total_characters = 0

total_tokens = 0

chunk_number = 0


with open(

    DATA_FILE,

    "r",

    encoding="utf-8",

    newline="\n"

) as input_file:


    with open(

        OUTPUT_FILE,

        "wb"

    ) as output_file:


        while True:


            # ====================================
            # READ NEXT TEXT CHUNK
            # ====================================

            text = input_file.read(
                CHUNK_CHARACTERS
            )


            if not text:

                break


            chunk_number += 1


            # ====================================
            # BPE TOKENIZATION
            # ====================================

            encoding = tokenizer.encode(
                text
            )


            token_ids = (
                encoding.ids
            )


            # ====================================
            # COMPACT TOKEN REPRESENTATION
            # ====================================

            token_array = np.asarray(

                token_ids,

                dtype=token_dtype
            )


            # ====================================
            # WRITE RAW TOKEN IDs
            # ====================================

            token_array.tofile(
                output_file
            )


            # ====================================
            # STATISTICS
            # ====================================

            total_characters += len(
                text
            )


            total_tokens += len(
                token_ids
            )


            elapsed = (

                time.perf_counter()

                -

                start_time
            )


            characters_per_token = (

                total_characters

                /

                total_tokens
            )


            tokens_per_second = (

                total_tokens

                /

                elapsed
            )


            output_size = (

                output_file.tell()

                /

                1024**2
            )


            print(

                f"chunk={chunk_number:3d} | "

                f"chars={total_characters:>13,} | "

                f"tokens={total_tokens:>12,} | "

                f"chars/token={characters_per_token:.3f} | "

                f"speed={tokens_per_second:>9,.0f} tok/s | "

                f"bin={output_size:>7.1f} MB"
            )


# ============================================================
# 7. FINAL STATISTICS
# ============================================================

end_time = (
    time.perf_counter()
)


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


tokens_per_second = (

    total_tokens

    /

    elapsed
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
    "Binary dataset:",
    round(
        binary_size / 1024**2,
        2
    ),
    "MB"
)


print(
    "Binary dataset:",
    round(
        binary_size / 1024**3,
        3
    ),
    "GB"
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
    "Average throughput:",
    f"{tokens_per_second:,.0f}",
    "tokens/sec"
)


# ============================================================
# 8. TRAIN / VALIDATION SPLIT
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
    "Train:",
    f"{TRAIN_FRACTION * 100:.0f}%"
)


print(
    "Validation:",
    f"{(1 - TRAIN_FRACTION) * 100:.0f}%"
)


# ============================================================
# 9. SAVE METADATA
# ============================================================

metadata = {

    "dataset_file":
        OUTPUT_FILE,

    "source_text_file":
        DATA_FILE,

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

    "tokens_per_character":
        tokens_per_character,

    "train_fraction":
        TRAIN_FRACTION,

    "train_tokens":
        train_tokens,

    "validation_tokens":
        validation_tokens,

    "chunk_characters":
        CHUNK_CHARACTERS
}


with open(

    METADATA_FILE,

    "w",

    encoding="utf-8",

    newline="\n"

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
# 10. VERIFY BINARY FILE SIZE
# ============================================================

expected_bytes_per_token = (

    np.dtype(
        token_dtype
    ).itemsize
)


expected_binary_size = (

    total_tokens

    *

    expected_bytes_per_token
)


print(
    "\nVERIFYING FILE SIZE"
)


print(
    "Expected bytes:",
    f"{expected_binary_size:,}"
)


print(
    "Actual bytes:",
    f"{binary_size:,}"
)


if (
    binary_size
    !=
    expected_binary_size
):

    raise RuntimeError(
        "Binary file size does not match "
        "the expected token count."
    )


print(
    "File-size verification: OK"
)


# ============================================================
# 11. MEMORY-MAP VERIFICATION
# ============================================================

print(
    "\nVERIFYING TOKEN ARRAY"
)


tokens = np.memmap(

    OUTPUT_FILE,

    dtype=token_dtype,

    mode="r"
)


print(
    "Memmap token count:",
    f"{len(tokens):,}"
)


print(
    "Expected token count:",
    f"{total_tokens:,}"
)


if (
    len(tokens)
    !=
    total_tokens
):

    raise RuntimeError(
        "Memmap token count mismatch."
    )


print(
    "Token-count verification: OK"
)


# ============================================================
# 12. TOKEN RANGE CHECK
# ============================================================

sample_check_size = min(

    1_000_000,

    len(tokens)
)


sample_tokens = tokens[
    :sample_check_size
]


minimum_token = int(
    sample_tokens.min()
)


maximum_token = int(
    sample_tokens.max()
)


print(
    "\nTOKEN RANGE CHECK"
)


print(
    "Minimum token ID:",
    minimum_token
)


print(
    "Maximum token ID:",
    maximum_token
)


print(
    "Vocabulary limit:",
    vocab_size - 1
)


if maximum_token >= vocab_size:

    raise RuntimeError(
        "Found token outside vocabulary."
    )


print(
    "Token range: OK"
)


# ============================================================
# 13. SAMPLE DECODE
# ============================================================

sample_size = min(

    150,

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


decoded_sample = (
    tokenizer.decode(
        sample_ids
    )
)


print(
    "\nFIRST 150 TOKENS DECODE TO:\n"
)


print(
    decoded_sample
)


# ============================================================
# 14. EFFECTIVE CONTEXT
# ============================================================

print(
    "\nAPPROXIMATE TEXT PER CONTEXT"
)


for context_length in [

    256,
    512,
    1024,
    2048,
    4096

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


# ============================================================
# 15. ESTIMATE TRAINING DATA SCALE
# ============================================================

print(
    "\nTRAINING SCALE FOR BLANK-0 v0.1"
)


BLANK0_PARAMETERS = (
    15_735_168
)


tokens_per_parameter = (

    train_tokens

    /

    BLANK0_PARAMETERS
)


print(
    "Blank-0 parameters:",
    f"{BLANK0_PARAMETERS:,}"
)


print(
    "Training tokens:",
    f"{train_tokens:,}"
)


print(
    "Unique training tokens / parameter:",
    round(
        tokens_per_parameter,
        2
    )
)


print(
    "\nBinary dataset verification: OK"
)

print(
    "\nDONE"
)