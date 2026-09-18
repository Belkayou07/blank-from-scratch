from datasets import load_dataset
import os
import re
import time


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_FILE = "blank_dataset_v1.txt"

TARGET_CHARACTERS = 100_000_000

MIN_DOCUMENT_LENGTH = 300

REPORT_EVERY = 5_000_000


# ============================================================
# CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""


    # Normalize Windows / old-Mac newlines.

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )


    # Remove NUL characters.

    text = text.replace(
        "\x00",
        ""
    )


    # Collapse huge runs of spaces/tabs,
    # while preserving normal paragraph structure.

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )


    # More than 3 consecutive newlines
    # becomes a normal paragraph break.

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )


    return text.strip()


# ============================================================
# LOAD FINEWEB-EDU AS A STREAM
# ============================================================

print(
    "Opening FineWeb-Edu stream..."
)


dataset = load_dataset(

    "HuggingFaceFW/fineweb-edu",

    name="sample-10BT",

    split="train",

    streaming=True
)


# ============================================================
# BUILD LOCAL DATASET
# ============================================================

print(
    f"Target: {TARGET_CHARACTERS:,} characters"
)

print(
    f"Output: {OUTPUT_FILE}\n"
)


characters_written = 0

documents_written = 0

documents_seen = 0

next_report = REPORT_EVERY

start_time = time.perf_counter()


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as output:

    for row in dataset:

        documents_seen += 1


        text = clean_text(
            row.get(
                "text",
                ""
            )
        )


        # Ignore extremely tiny fragments.

        if (
            len(text)
            <
            MIN_DOCUMENT_LENGTH
        ):

            continue


        remaining = (
            TARGET_CHARACTERS
            -
            characters_written
        )


        if remaining <= 0:
            break


        # ----------------------------------------
        # Do not exceed target size massively.
        # ----------------------------------------

        if len(text) > remaining:

            text = text[
                :remaining
            ]


        # ----------------------------------------
        # Separate documents clearly.
        # ----------------------------------------

        if documents_written > 0:

            separator = "\n\n"

            output.write(
                separator
            )

            characters_written += len(
                separator
            )


        output.write(
            text
        )


        characters_written += len(
            text
        )


        documents_written += 1


        # ----------------------------------------
        # Progress
        # ----------------------------------------

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


            print(
                f"{characters_written:>12,} chars | "
                f"{documents_written:>7,} docs | "
                f"{elapsed:8.1f} sec"
            )


            next_report += (
                REPORT_EVERY
            )


        if (
            characters_written
            >=
            TARGET_CHARACTERS
        ):

            break


# ============================================================
# FINAL REPORT
# ============================================================

elapsed = (
    time.perf_counter()
    -
    start_time
)


file_size = os.path.getsize(
    OUTPUT_FILE
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
    "Characters:",
    f"{characters_written:,}"
)


print(
    "File size:",
    round(
        file_size
        /
        1024**2,
        2
    ),
    "MB"
)


print(
    "Time:",
    round(
        elapsed,
        2
    ),
    "seconds"
)


print(
    "\nSaved as:",
    OUTPUT_FILE
)