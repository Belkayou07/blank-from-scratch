import random
from collections import defaultdict, Counter


# ==================================================
# TRAINING TEXT
# ==================================================

text = """
the cat sleeps on the mat.
the cat walks through the house.
the dog sleeps near the door.
the dog walks through the garden.
the cat sees the dog.
the dog sees the cat.
""".strip()


# ==================================================
# CONTEXT LENGTH
# ==================================================

context_length = 3


# ==================================================
# LEARN:
#
# last 3 characters -> next character
# ==================================================

transitions = defaultdict(Counter)


for i in range(len(text) - context_length):

    context = text[i:i + context_length]

    next_character = text[i + context_length]

    transitions[context][next_character] += 1


print("Number of different contexts:", len(transitions))


# ==================================================
# INSPECT SOME CONTEXTS
# ==================================================

examples = [
    "the",
    " ca",
    " do",
    "wal",
]


print("\nSOME LEARNED CONTEXTS:")

for context in examples:

    print(
        repr(context),
        "→",
        dict(transitions.get(context, {}))
    )


# ==================================================
# GENERATION
# ==================================================

current_context = "the"

generated = current_context


for _ in range(200):

    possibilities = transitions.get(current_context)

    if not possibilities:
        break


    characters = list(possibilities.keys())
    counts = list(possibilities.values())


    next_character = random.choices(
        characters,
        weights=counts,
        k=1
    )[0]


    generated += next_character


    # Keep ONLY the last 3 characters.
    current_context = (
        current_context + next_character
    )[-context_length:]


print("\nGENERATED TEXT:\n")
print(generated)