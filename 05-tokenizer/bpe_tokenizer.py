import json
import os
from collections import Counter


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = "tinyshakespeare.txt"

TOKENIZER_FILE = "bpe_tokenizer.json"


# The first 256 tokens are raw byte values.
#
# 0 ... 255
#
# Everything above that will be learned BPE merges.

BASE_VOCAB_SIZE = 256


# Educational tokenizer.
#
# 384 - 256 = 128 learned merges.

TARGET_VOCAB_SIZE = 384


# We don't need the entire corpus just to demonstrate
# how BPE learning works.

TRAINING_CHARACTERS = 250_000


# ============================================================
# 2. READ CORPUS
# ============================================================

if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        f"{DATA_FILE} was not found."
    )


with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    full_text = f.read()


training_text = full_text[
    :TRAINING_CHARACTERS
]


print(
    "Tokenizer training characters:",
    len(training_text)
)


# ============================================================
# 3. TEXT -> RAW BYTES
# ============================================================

raw_bytes = training_text.encode(
    "utf-8"
)


# bytes(...) contains integers from 0 through 255.

tokens = list(raw_bytes)


print(
    "Initial byte tokens:",
    len(tokens)
)


# ============================================================
# 4. MERGE FUNCTION
# ============================================================

def merge_pair(
    token_ids,
    pair,
    new_token_id
):

    """
    Replace every occurrence of:

        (A, B)

    with:

        new_token_id
    """

    output = []

    i = 0


    while i < len(token_ids):

        # Check whether the next two tokens
        # match the pair we want to merge.

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


# ============================================================
# 5. TRAIN BPE
# ============================================================

merges = []


number_of_merges = (
    TARGET_VOCAB_SIZE
    -
    BASE_VOCAB_SIZE
)


print(
    "\nLearning",
    number_of_merges,
    "BPE merges...\n"
)


for merge_index in range(
    number_of_merges
):

    # ----------------------------------------
    # Count adjacent token pairs
    # ----------------------------------------

    pair_counts = Counter(
        zip(
            tokens,
            tokens[1:]
        )
    )


    if not pair_counts:

        break


    # ----------------------------------------
    # Most common pair
    # ----------------------------------------

    best_pair, count = (
        pair_counts.most_common(1)[0]
    )


    # ----------------------------------------
    # Allocate a new token ID
    # ----------------------------------------

    new_token_id = (
        BASE_VOCAB_SIZE
        +
        merge_index
    )


    # ----------------------------------------
    # Record merge
    # ----------------------------------------

    merges.append(
        (
            best_pair[0],
            best_pair[1],
            new_token_id
        )
    )


    # ----------------------------------------
    # Apply merge to training sequence
    # ----------------------------------------

    tokens = merge_pair(
        tokens,
        best_pair,
        new_token_id
    )


    if (
        merge_index % 16 == 0
        or
        merge_index
        == number_of_merges - 1
    ):

        print(
            f"merge={merge_index + 1:3d} | "
            f"pair={best_pair} | "
            f"count={count:6d} | "
            f"tokens remaining={len(tokens)}"
        )


# ============================================================
# 6. BUILD TOKEN VOCABULARY
# ============================================================
#
# vocab[token_id] = bytes represented by that token

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
    "\nFinal vocabulary size:",
    len(vocab)
)


# ============================================================
# 7. ENCODE
# ============================================================

def encode(text):

    """
    Convert normal text into BPE token IDs.
    """

    token_ids = list(
        text.encode("utf-8")
    )


    # Apply merges in the exact order
    # in which they were learned.

    for (
        left,
        right,
        new_id
    ) in merges:

        token_ids = merge_pair(
            token_ids,
            (left, right),
            new_id
        )


    return token_ids


# ============================================================
# 8. DECODE
# ============================================================

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
# 9. SAVE TOKENIZER
# ============================================================

tokenizer_data = {

    "base_vocab_size":
        BASE_VOCAB_SIZE,

    "target_vocab_size":
        TARGET_VOCAB_SIZE,

    "merges": [
        [left, right, new_id]

        for left, right, new_id
        in merges
    ]
}


with open(
    TOKENIZER_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        tokenizer_data,
        f,
        indent=2
    )


print(
    "\nTokenizer saved:",
    TOKENIZER_FILE
)


# ============================================================
# 10. INSPECT LEARNED TOKENS
# ============================================================

print(
    "\nSOME LEARNED TOKENS:\n"
)


for token_id in range(
    BASE_VOCAB_SIZE,
    min(
        BASE_VOCAB_SIZE + 40,
        len(vocab)
    )
):

    represented_bytes = vocab[
        token_id
    ]


    represented_text = (
        represented_bytes.decode(
            "utf-8",
            errors="replace"
        )
    )


    print(
        f"{token_id:3d} → "
        f"{represented_text!r}"
    )


# ============================================================
# 11. TEST ROUND TRIP
# ============================================================

sample = (
    "ROMEO: Where is the king?"
)


encoded = encode(
    sample
)


decoded = decode(
    encoded
)


print(
    "\nSAMPLE:"
)

print(
    sample
)


print(
    "\nENCODED:"
)

print(
    encoded
)


print(
    "\nNUMBER OF TOKENS:",
    len(encoded)
)


print(
    "\nDECODED:"
)

print(
    decoded
)


print(
    "\nROUND TRIP CORRECT:",
    decoded == sample
)


# ============================================================
# 12. COMPRESSION TEST
# ============================================================

comparison_text = full_text[
    :50_000
]


character_count = len(
    comparison_text
)


byte_count = len(
    comparison_text.encode(
        "utf-8"
    )
)


bpe_tokens = encode(
    comparison_text
)


bpe_count = len(
    bpe_tokens
)


print(
    "\nCOMPRESSION TEST"
)


print(
    "Characters:",
    character_count
)


print(
    "Raw byte tokens:",
    byte_count
)


print(
    "BPE tokens:",
    bpe_count
)


print(
    "Characters per BPE token:",
    round(
        character_count
        /
        bpe_count,
        3
    )
)


print(
    "Token reduction vs byte-level:",
    f"{(
        1
        -
        bpe_count / byte_count
    ) * 100:.2f}%"
)