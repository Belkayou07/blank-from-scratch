import os
import time

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = "blank_dataset_v1.txt"

TOKENIZER_FILE = "blank_tokenizer_v1.json"

VOCAB_SIZE = 4096

MIN_FREQUENCY = 2


# ============================================================
# 2. CHECK DATASET
# ============================================================

if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        f"{DATA_FILE} was not found."
    )


file_size = os.path.getsize(
    DATA_FILE
)


print(
    "Dataset:",
    DATA_FILE
)

print(
    "Dataset size:",
    round(
        file_size / 1024**2,
        2
    ),
    "MB"
)


# ============================================================
# 3. CREATE AN EMPTY BPE TOKENIZER
# ============================================================
#
# IMPORTANT:
#
# This tokenizer knows NOTHING yet.
#
# BPE()
#
# contains no pretrained vocabulary
# and no pretrained merge rules.

tokenizer = Tokenizer(
    BPE()
)


# ============================================================
# 4. BYTE-LEVEL PRE-TOKENIZER
# ============================================================
#
# ByteLevel maps every possible byte to one of
# 256 base symbols.
#
# Therefore arbitrary UTF-8 text can always
# be represented, even if a language was not
# present during tokenizer training.

tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False
)


# Decoder reverses the byte-level representation
# back into normal text.

tokenizer.decoder = (
    ByteLevelDecoder()
)


# ============================================================
# 5. BPE TRAINER
# ============================================================

trainer = BpeTrainer(

    vocab_size=VOCAB_SIZE,

    min_frequency=MIN_FREQUENCY,

    show_progress=True,

    # Force all 256 byte symbols into
    # the base vocabulary.
    initial_alphabet=(
        ByteLevel.alphabet()
    )
)


# ============================================================
# 6. TRAIN TOKENIZER
# ============================================================

print(
    "\nTRAINING TOKENIZER"
)

print(
    "Target vocabulary:",
    f"{VOCAB_SIZE:,}"
)


start = time.perf_counter()


tokenizer.train(
    [DATA_FILE],
    trainer
)


end = time.perf_counter()


print(
    "\nTokenizer training time:",
    round(
        end - start,
        2
    ),
    "seconds"
)


# ============================================================
# 7. SAVE TOKENIZER
# ============================================================

tokenizer.save(
    TOKENIZER_FILE
)


print(
    "Tokenizer saved:",
    TOKENIZER_FILE
)


# ============================================================
# 8. ACTUAL VOCABULARY SIZE
# ============================================================

actual_vocab_size = (
    tokenizer.get_vocab_size()
)


print(
    "Actual vocabulary size:",
    f"{actual_vocab_size:,}"
)


# ============================================================
# 9. ROUND-TRIP TEST
# ============================================================

sample = (
    "The king returned to Brussels. "
    "Café, français, Nederlands — 日本語."
)


encoding = tokenizer.encode(
    sample
)


decoded = tokenizer.decode(
    encoding.ids
)


print(
    "\nROUND-TRIP TEST"
)


print(
    "\nOriginal:"
)

print(
    sample
)


print(
    "\nToken IDs:"
)

print(
    encoding.ids
)


print(
    "\nNumber of tokens:",
    len(encoding.ids)
)


print(
    "\nDecoded:"
)

print(
    decoded
)


print(
    "\nExact round trip:",
    decoded == sample
)


# ============================================================
# 10. SHOW TOKEN PIECES
# ============================================================

print(
    "\nTOKEN PIECES:\n"
)


for token_id in encoding.ids:

    piece = tokenizer.decode(
        [token_id]
    )


    print(
        f"{token_id:4d} -> "
        f"{piece!r}"
    )


# ============================================================
# 11. LARGE COMPRESSION TEST
# ============================================================
#
# Read the first one million characters
# from our actual training corpus.

TEST_CHARACTERS = 1_000_000


with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    comparison_text = f.read(
        TEST_CHARACTERS
    )


comparison_bytes = (
    comparison_text.encode(
        "utf-8"
    )
)


print(
    "\nEncoding compression sample..."
)


compression_start = (
    time.perf_counter()
)


comparison_encoding = (
    tokenizer.encode(
        comparison_text
    )
)


compression_end = (
    time.perf_counter()
)


token_count = len(
    comparison_encoding.ids
)


character_count = len(
    comparison_text
)


byte_count = len(
    comparison_bytes
)


characters_per_token = (
    character_count
    /
    token_count
)


bytes_per_token = (
    byte_count
    /
    token_count
)


reduction_vs_bytes = (

    1
    -
    token_count
    /
    byte_count

) * 100


print(
    "\nCOMPRESSION RESULTS"
)


print(
    "Characters:",
    f"{character_count:,}"
)


print(
    "UTF-8 bytes:",
    f"{byte_count:,}"
)


print(
    "BPE tokens:",
    f"{token_count:,}"
)


print(
    "Characters/token:",
    round(
        characters_per_token,
        3
    )
)


print(
    "Bytes/token:",
    round(
        bytes_per_token,
        3
    )
)


print(
    "Reduction vs byte tokens:",
    f"{reduction_vs_bytes:.2f}%"
)


print(
    "Encoding time:",
    round(
        compression_end
        -
        compression_start,
        3
    ),
    "seconds"
)


# ============================================================
# 12. CONTEXT INTERPRETATION
# ============================================================

print(
    "\nEFFECTIVE TEXT CONTEXT"
)


for context_tokens in [
    256,
    512,
    1024
]:

    estimated_characters = (

        context_tokens
        *
        characters_per_token
    )


    print(
        f"{context_tokens:4d} tokens "
        f"≈ {estimated_characters:,.0f} "
        f"characters"
    )


# ============================================================
# 13. INSPECT LEARNED VOCABULARY
# ============================================================

vocab = tokenizer.get_vocab()


id_to_token = {

    token_id: token

    for token, token_id
    in vocab.items()
}


print(
    "\nSAMPLE LEARNED TOKENS:\n"
)


shown = 0


# Skip most of the raw byte alphabet
# and inspect later BPE tokens.

for token_id in range(
    256,
    actual_vocab_size
):

    if token_id not in id_to_token:
        continue


    decoded_piece = tokenizer.decode(
        [token_id]
    )


    # Show useful multi-character pieces.

    if len(decoded_piece) >= 2:

        print(
            f"{token_id:4d} -> "
            f"{decoded_piece!r}"
        )


        shown += 1


        if shown >= 50:
            break


print(
    "\nDONE"
)