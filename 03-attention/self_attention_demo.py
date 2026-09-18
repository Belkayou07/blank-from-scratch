import math
import torch
import torch.nn.functional as F


torch.manual_seed(7)


# ==================================================
# 1. FOUR TOKENS
# ==================================================

tokens = [
    "the",
    "cat",
    "sat",
    "down"
]

T = len(tokens)


# ==================================================
# 2. TOKEN REPRESENTATIONS
# ==================================================
#
# For this demonstration we use simple one-hot vectors.
#
# "the"  = [1, 0, 0, 0]
# "cat"  = [0, 1, 0, 0]
# "sat"  = [0, 0, 1, 0]
# "down" = [0, 0, 0, 1]
#
# Later these will be learned embeddings.

X = torch.eye(4)

d_model = 4
d_k = 4


print("TOKEN REPRESENTATIONS X:")
print(X)


# ==================================================
# 3. Q, K, V PROJECTION MATRICES
# ==================================================
#
# These are normally learned parameters.
#
# They start randomly in a blank Transformer.

W_Q = torch.randn(d_model, d_k) * 0.5
W_K = torch.randn(d_model, d_k) * 0.5
W_V = torch.randn(d_model, d_k) * 0.5


# Every token creates:
#
# Query = what am I looking for?
# Key   = what kind of information do I contain?
# Value = what information can I contribute?

Q = X @ W_Q
K = X @ W_K
V = X @ W_V


print("\nQUERIES Q:")
print(Q.round(decimals=3))

print("\nKEYS K:")
print(K.round(decimals=3))

print("\nVALUES V:")
print(V.round(decimals=3))


# ==================================================
# 4. ATTENTION SCORES
# ==================================================
#
# Compare every Query against every Key.
#
# Q @ K.T gives:
#
#              KEY
#          the cat sat down
#
# QUERY the  ...
#       cat  ...
#       sat  ...
#       down ...
#
# Each row asks:
#
# "How relevant is every token to me?"

scores = Q @ K.T


# Scale scores.
scores = scores / math.sqrt(d_k)


print("\nRAW ATTENTION SCORES:")
print(scores.round(decimals=3))


# ==================================================
# 5. CAUSAL MASK
# ==================================================
#
# We are building a GPT-style decoder.
#
# A token may look at:
#
# itself
# +
# previous tokens
#
# but NOT future tokens.

mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool),
    diagonal=1
)


masked_scores = scores.masked_fill(
    mask,
    float("-inf")
)


print("\nAFTER CAUSAL MASK:")
print(masked_scores.round(decimals=3))


# ==================================================
# 6. SOFTMAX
# ==================================================
#
# Turn scores into attention probabilities.

attention = F.softmax(
    masked_scores,
    dim=-1
)


print("\nATTENTION WEIGHTS:")

for i, token in enumerate(tokens):

    print(f"\n{token!r} looks at:")

    for j, other_token in enumerate(tokens):

        probability = attention[i, j].item()

        print(
            f"    {other_token!r:6s} "
            f"{probability * 100:6.2f}%"
        )


# ==================================================
# 7. USE VALUES
# ==================================================
#
# Weighted combination of the Value vectors.

output = attention @ V


print("\nFINAL CONTEXTUAL REPRESENTATIONS:")
print(output.round(decimals=3))