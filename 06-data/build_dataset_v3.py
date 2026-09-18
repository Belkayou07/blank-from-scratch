import gc
import json
import os
import time

from datasets import load_dataset


# ============================================================
# CONFIG
# ============================================================

OUTPUT_FILE = "blank_dataset_v3.txt"
METADATA_FILE = "blank_dataset_v3_meta.json"

TARGET_CHARACTERS = 3_000_000_000

MIN_DOCUMENT_LENGTH = 300

PROGRESS_EVERY = 100_000_000


# ============================================================
# SAFETY
# ============================================================

if os.path.exists(OUTPUT_FILE):
    raise FileExistsError(
        f"{OUTPUT_FILE} already exists.\n"
        "Delete or rename it before building Dataset v3."
    )


print("BLANK DATASET v3")
print("================")
print(f"Target characters: {TARGET_CHARACTERS:,}")
print(f"Minimum document length: {MIN_DOCUMENT_LENGTH:,}")
print()
print("Source: FineWeb-Edu sample-10BT")
print("Streaming: enabled")
print()


# ============================================================
# LOAD STREAMING DATASET
# ============================================================

print("Opening FineWeb-Edu stream...")


dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)


# ============================================================
# BUILD DATASET
# ============================================================

start_time = time.perf_counter()

characters_written = 0

documents_seen = 0

documents_written = 0

documents_skipped = 0

next_progress = PROGRESS_EVERY


# newline="\n" prevents Windows from turning \n into \r\n.
#
# That keeps our Python character count consistent with the
# physical text we intend to tokenize later.

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline="\n",
) as output:

    for example in dataset:

        documents_seen += 1


        text = example.get(
            "text",
            ""
        )


        if not isinstance(text, str):

            documents_skipped += 1
            continue


        # ----------------------------------------
        # Normalize line endings
        # ----------------------------------------

        text = text.replace(
            "\r\n",
            "\n"
        )

        text = text.replace(
            "\r",
            "\n"
        )


        # Remove only outer whitespace.
        #
        # We deliberately preserve internal
        # formatting/newlines.

        text = text.strip()


        if len(text) < MIN_DOCUMENT_LENGTH:

            documents_skipped += 1
            continue


        # ----------------------------------------
        # Separator
        # ----------------------------------------

        if documents_written > 0:

            separator = "\n\n"

        else:

            separator = ""


        remaining = (
            TARGET_CHARACTERS
            -
            characters_written
        )


        if remaining <= 0:
            break


        # ----------------------------------------
        # Complete document fits
        # ----------------------------------------

        required = (
            len(separator)
            +
            len(text)
        )


        if required <= remaining:

            output.write(
                separator
            )

            output.write(
                text
            )

            characters_written += (
                required
            )

            documents_written += 1


        # ----------------------------------------
        # Final partial document
        # ----------------------------------------

        else:

            if separator:

                if remaining <= len(separator):

                    output.write(
                        separator[
                            :remaining
                        ]
                    )

                    characters_written += (
                        remaining
                    )

                    break


                output.write(
                    separator
                )

                characters_written += (
                    len(separator)
                )

                remaining -= (
                    len(separator)
                )


            if remaining > 0:

                output.write(
                    text[
                        :remaining
                    ]
                )

                characters_written += (
                    remaining
                )

                documents_written += 1


            break


        # ----------------------------------------
        # Progress
        # ----------------------------------------

        if characters_written >= next_progress:

            elapsed = (
                time.perf_counter()
                -
                start_time
            )


            percent = (

                characters_written

                /

                TARGET_CHARACTERS

                *

                100
            )


            speed = (

                characters_written

                /

                max(
                    elapsed,
                    0.001
                )
            )


            print(
                f"{characters_written:,} / "
                f"{TARGET_CHARACTERS:,} chars "
                f"({percent:.1f}%) | "
                f"docs={documents_written:,} | "
                f"{speed / 1_000_000:.2f} M chars/sec"
            )


            while (
                next_progress
                <=
                characters_written
            ):

                next_progress += (
                    PROGRESS_EVERY
                )


# ============================================================
# FINISH
# ============================================================

elapsed = (
    time.perf_counter()
    -
    start_time
)


file_size_bytes = os.path.getsize(
    OUTPUT_FILE
)


file_size_mb = (

    file_size_bytes

    /

    1024**2
)


file_size_gb = (

    file_size_bytes

    /

    1024**3
)


# ============================================================
# VERIFY EXACT CHARACTER COUNT
# ============================================================
#
# Read it back in chunks so we verify the final file itself,
# not merely our counter.
# ============================================================

print()
print("Verifying final character count...")


verified_characters = 0


with open(
    OUTPUT_FILE,
    "r",
    encoding="utf-8",
    newline="",
) as f:

    while True:

        chunk = f.read(
            8_000_000
        )


        if not chunk:
            break


        verified_characters += len(
            chunk
        )


if verified_characters != TARGET_CHARACTERS:

    raise RuntimeError(
        f"Character verification failed: "
        f"expected {TARGET_CHARACTERS:,}, "
        f"found {verified_characters:,}"
    )


# ============================================================
# METADATA
# ============================================================

metadata = {

    "dataset_name":
        "Blank Dataset v3",

    "source":
        "HuggingFaceFW/fineweb-edu",

    "source_config":
        "sample-10BT",

    "target_characters":
        TARGET_CHARACTERS,

    "verified_characters":
        verified_characters,

    "minimum_document_length":
        MIN_DOCUMENT_LENGTH,

    "documents_seen":
        documents_seen,

    "documents_written":
        documents_written,

    "documents_skipped":
        documents_skipped,

    "file_size_bytes":
        file_size_bytes,

    "build_seconds":
        elapsed,
}


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 70)
print("DATASET v3 COMPLETE")
print("=" * 70)

print(
    "Characters:",
    f"{verified_characters:,}"
)

print(
    "Documents seen:",
    f"{documents_seen:,}"
)

print(
    "Documents written:",
    f"{documents_written:,}"
)

print(
    "Documents skipped:",
    f"{documents_skipped:,}"
)

print(
    "File size:",
    f"{file_size_mb:,.2f} MB"
)

print(
    "File size:",
    f"{file_size_gb:.3f} GB"
)

print(
    "Build time:",
    f"{elapsed:.2f} seconds"
)

print(
    "Text file:",
    OUTPUT_FILE
)

print(
    "Metadata:",
    METADATA_FILE
)

print()
print("Verification: PASSED")
print("DONE")


# Encourage streaming/network objects to close cleanly.
del dataset

gc.collect()