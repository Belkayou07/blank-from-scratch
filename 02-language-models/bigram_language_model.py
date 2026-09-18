import random
from collections import defaultdict, Counter


# ==================================================
# TRAINING DATA
# ==================================================

text = """
the cat sleeps on the mat.
the cat walks through the house.
the dog sleeps near the door.
the dog walks through the garden.
the cat sees the dog.
the dog sees the cat.
"""


# Remove the initial/final whitespace from
# the triple-quoted Python string.
text = text.strip()


# ==================================================
# VOCABULARY
# ==================================================

characters = sorted(set(text))

print("Vocabulary:")
print(characters)

print("\nVocabulary size:", len(characters))


# ==================================================
# LEARN CHARACTER TRANSITIONS
# ==================================================

transitions = defaultdict(Counter)


for current_char, next_char in zip(text, text[1:]):

    transitions[current_char][next_char] += 1


# ==================================================
# INSPECT WHAT THE MODEL LEARNED
# ==================================================

print("\nWhat follows 't'?")

for character, count in transitions["t"].items():

    print(
        repr(character),
        "count:",
        count
    )


# ==================================================
# TURN COUNTS INTO PROBABILISTIC CHOICES
# ==================================================

def choose_next_character(current_char):

    possible_next = transitions[current_char]

    characters = list(possible_next.keys())
    counts = list(possible_next.values())

    return random.choices(
        characters,
        weights=counts,
        k=1
    )[0]


# ==================================================
# GENERATE TEXT
# ==================================================

current = "t"

generated = current


for _ in range(200):

    if current not in transitions:
        break

    next_char = choose_next_character(current)

    generated += next_char

    current = next_char


print("\nGENERATED TEXT:\n")
print(generated)