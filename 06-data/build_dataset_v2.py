import json
import os
import re
import time

from datasets import load_dataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

OUTPUT_FILE = "blank_dataset_v2.txt"

METADATA_FILE = "blank_dataset_v2_meta.json"


TARGET_CHARACTERS = 1_000_000_000

MIN_DOCUMENT_LENGTH = 300

REPORT_EVERY = 50_000_000


DATASET_NAME = (
    "HuggingFaceFW/fineweb-edu"
)

DATASET_CONFIG = (
    "sample-10BT"
)


# ============================================================
# 2. CLEANING
# ============================================================

def clean_text(text):

    if not text:

        return ""


    # ----------------------------------------
    # Normalize newlines inside the document
    # ----------------------------------------

    text = text.replace(
        "\r\n",
        "\n"
    )


    text = text.replace(
        "\r",
        "\n"
    )


    # ----------------------------------------
    # Remove NUL characters
    # ----------------------------------------

    text = text.replace(
        "\x00",
        ""
    )


    # ----------------------------------------
    # Collapse repeated spaces/tabs
    # ----------------------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )


    # ----------------------------------------
    # Prevent enormous vertical gaps
    # ----------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )


    return text.strip()


# ============================================================
# 3. OPEN STREAM
# ============================================================

print(
    "Opening FineWeb-Edu stream..."
)


dataset = load_dataset(

    DATASET_NAME,

    name=DATASET_CONFIG,

    split="train",

    streaming=True
)


print(
    f"Target: {TARGET_CHARACTERS:,} characters"
)

print(
    "Output:",
    OUTPUT_FILE
)


# ============================================================
# 4. BUILD DATASET
# ============================================================

characters_written = 0

documents_seen = 0

documents_written = 0

documents_skipped = 0


next_report = (
    REPORT_EVERY
)


start_time = (
    time.perf_counter()
)


# IMPORTANT:
#
# newline="\n"
#
# prevents Windows from silently turning:
#
# \n
#
# into:
#
# \r\n
#
# This means our character accounting now matches
# the actual text that the tokenizer later reads.

with open(

    OUTPUT_FILE,

    "w",

    encoding="utf-8",

    newline="\n"

) as output:


    for row in dataset:

        documents_seen += 1


        # ========================================
        # EXTRACT TEXT
        # ========================================

        text = clean_text(

            row.get(
                "text",
                ""
            )
        )


        # ========================================
        # SKIP TINY DOCUMENTS
        # ========================================

        if (
            len(text)
            <
            MIN_DOCUMENT_LENGTH
        ):

            documents_skipped += 1

            continue


        # ========================================
        # DOCUMENT SEPARATOR
        # ========================================

        if documents_written > 0:

            separator = "\n\n"


            remaining = (

                TARGET_CHARACTERS

                -

                characters_written
            )


            # If only 1 or 2 characters remain,
            # write only what is needed.

            if remaining <= len(
                separator
            ):

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


            characters_written += len(
                separator
            )


        # ========================================
        # REMAINING CHARACTER BUDGET
        # ========================================

        remaining = (

            TARGET_CHARACTERS

            -

            characters_written
        )


        if remaining <= 0:

            break


        # ========================================
        # TRIM FINAL DOCUMENT IF NECESSARY
        # ========================================

        if len(text) > remaining:

            text = text[
                :remaining
            ]


        # ========================================
        # WRITE DOCUMENT
        # ========================================

        output.write(
            text
        )


        characters_written += len(
            text
        )


        documents_written += 1


        # ========================================
        # PROGRESS REPORT
        # ========================================

        if (
            characters_written
            >=
            next_report
        ):

            elapsed = (

                time.perf_counter()

                -

                start_time
            )


            file_size = os.path.getsize(
                OUTPUT_FILE
            )


            print(

                f"{characters_written:>15,} chars | "

                f"{documents_written:>8,} docs | "

                f"{file_size / 1024**2:>8.1f} MB | "

                f"{elapsed:>8.1f} sec"
            )


            while (
                next_report
                <=
                characters_written
            ):

                next_report += (
                    REPORT_EVERY
                )


        # ========================================
        # EXACT TARGET REACHED
        # ========================================

        if (
            characters_written
            >=
            TARGET_CHARACTERS
        ):

            break


# ============================================================
# 5. FINAL STATISTICS
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


print(
    "\nDATASET COMPLETE"
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
    "Characters:",
    f"{characters_written:,}"
)


print(
    "Target characters:",
    f"{TARGET_CHARACTERS:,}"
)


print(
    "Exact target:",
    (
        characters_written
        ==
        TARGET_CHARACTERS
    )
)


print(
    "File size:",
    round(
        file_size_mb,
        2
    ),
    "MB"
)


print(
    "File size:",
    round(
        file_size_gb,
        3
    ),
    "GB"
)


print(
    "Build time:",
    round(
        elapsed,
        2
    ),
    "seconds"
)


# ============================================================
# 6. SAFETY CHECK
# ============================================================

if (
    characters_written
    !=
    TARGET_CHARACTERS
):

    raise RuntimeError(

        "Dataset did not reach the "
        "requested target size."
    )


# ============================================================
# 7. SAVE METADATA
# ============================================================

metadata = {

    "dataset_name":
        DATASET_NAME,

    "dataset_config":
        DATASET_CONFIG,

    "output_file":
        OUTPUT_FILE,

    "target_characters":
        TARGET_CHARACTERS,

    "characters_written":
        characters_written,

    "documents_seen":
        documents_seen,

    "documents_written":
        documents_written,

    "documents_skipped":
        documents_skipped,

    "minimum_document_length":
        MIN_DOCUMENT_LENGTH,

    "file_size_bytes":
        file_size_bytes,

    "newline_format":
        "LF",

    "encoding":
        "UTF-8"
}


with open(

    METADATA_FILE,

    "w",

    encoding="utf-8",

    newline="\n"

) as f:

    json.dump(

        metadata,

        f,

        indent=2
    )


print(
    "\nMetadata saved:",
    METADATA_FILE
)


# ============================================================
# 8. QUICK VERIFICATION
# ============================================================

print(
    "\nVERIFYING DATASET"
)


with open(

    OUTPUT_FILE,

    "r",

    encoding="utf-8",

    newline="\n"

) as f:

    beginning = f.read(
        1000
    )


print(
    "\nFIRST 1000 CHARACTERS:\n"
)


print(
    beginning
)


print(
    "\nSaved as:",
    OUTPUT_FILE
)


print(
    "\nDONE"
)