import math
import random


# ==================================================
# OUR AUTOGRAD ENGINE
# ==================================================

class Value:

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            (self, other),
            "+"
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    __radd__ = __add__


    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data * other.data,
            (self, other),
            "*"
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    __rmul__ = __mul__


    def __neg__(self):
        return self * -1


    def __sub__(self, other):
        return self + (-other)


    def __rsub__(self, other):
        return other + (-self)


    def __pow__(self, exponent):

        out = Value(
            self.data ** exponent,
            (self,),
            f"**{exponent}"
        )

        def _backward():
            self.grad += (
                exponent
                * self.data ** (exponent - 1)
                * out.grad
            )

        out._backward = _backward
        return out


    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        return self * (other ** -1)


    def exp(self):

        result = math.exp(self.data)

        out = Value(
            result,
            (self,),
            "exp"
        )

        def _backward():
            self.grad += result * out.grad

        out._backward = _backward
        return out


    def log(self):

        out = Value(
            math.log(self.data),
            (self,),
            "log"
        )

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out


    def backward(self):

        topo = []
        visited = set()

        def build_topology(value):

            if value not in visited:

                visited.add(value)

                for child in value._prev:
                    build_topology(child)

                topo.append(value)

        build_topology(self)

        self.grad = 1.0

        for value in reversed(topo):
            value._backward()



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
# TOKENIZER
# ==================================================

vocabulary = sorted(set(text))

vocab_size = len(vocabulary)


# character -> integer
stoi = {
    character: index
    for index, character in enumerate(vocabulary)
}


# integer -> character
itos = {
    index: character
    for character, index in stoi.items()
}


print("Vocabulary size:", vocab_size)



# ==================================================
# CONVERT TEXT INTO TOKEN IDs
# ==================================================

tokens = [
    stoi[character]
    for character in text
]


# Training pairs:
#
# current character -> next character

pairs = list(
    zip(tokens, tokens[1:])
)

print("Training pairs:", len(pairs))



# ==================================================
# MODEL PARAMETERS
# ==================================================
#
# W[current_character][possible_next_character]
#
# 20 x 20 = 400 parameters

random.seed(42)

W = [

    [
        Value(random.uniform(-0.1, 0.1))
        for _ in range(vocab_size)
    ]

    for _ in range(vocab_size)
]


parameters = [
    parameter
    for row in W
    for parameter in row
]


print("Model parameters:", len(parameters))



# ==================================================
# SOFTMAX
# ==================================================

def softmax(logits):

    # Numerical stability:
    # subtracting the largest score doesn't change
    # the final softmax probabilities.

    maximum = max(
        logit.data
        for logit in logits
    )

    exponentials = [
        (logit - maximum).exp()
        for logit in logits
    ]

    total = sum(exponentials)

    probabilities = [
        value / total
        for value in exponentials
    ]

    return probabilities



# ==================================================
# TRAINING
# ==================================================

learning_rate = 5.0
epochs = 300


for epoch in range(epochs):

    losses = []


    # ----------------------------------------------
    # FORWARD PASS
    # ----------------------------------------------

    for current_token, target_token in pairs:

        # Select 20 learnable scores belonging
        # to the current character.
        logits = W[current_token]


        # Convert scores into probabilities.
        probabilities = softmax(logits)


        # Probability assigned to the actual
        # correct next character.
        correct_probability = probabilities[target_token]


        # Cross-entropy loss for this prediction.
        loss = -correct_probability.log()

        losses.append(loss)


    # Average loss across the whole training corpus.
    total_loss = sum(losses)

    loss = total_loss / len(pairs)


    # ----------------------------------------------
    # CLEAR OLD GRADIENTS
    # ----------------------------------------------

    for parameter in parameters:
        parameter.grad = 0.0


    # ----------------------------------------------
    # BACKPROPAGATION
    # ----------------------------------------------

    loss.backward()


    # ----------------------------------------------
    # GRADIENT DESCENT
    # ----------------------------------------------

    for parameter in parameters:

        parameter.data -= (
            learning_rate
            * parameter.grad
        )


    if epoch % 50 == 0:

        print(
            f"epoch={epoch:3d} | "
            f"loss={loss.data:.6f}"
        )



# ==================================================
# INSPECT LEARNED PROBABILITIES
# ==================================================

def get_probabilities(character):

    token_id = stoi[character]

    raw_logits = [
        value.data
        for value in W[token_id]
    ]

    maximum = max(raw_logits)

    exponentials = [
        math.exp(value - maximum)
        for value in raw_logits
    ]

    total = sum(exponentials)

    return [
        value / total
        for value in exponentials
    ]


print("\nLEARNED PROBABILITIES AFTER 't':")

probabilities = get_probabilities("t")


results = [

    (
        itos[index],
        probability
    )

    for index, probability
    in enumerate(probabilities)
]


results.sort(
    key=lambda item: item[1],
    reverse=True
)


for character, probability in results[:5]:

    print(
        repr(character),
        f"{probability * 100:.2f}%"
    )



# ==================================================
# GENERATE TEXT
# ==================================================

current_character = "t"

generated = current_character


for _ in range(200):

    probabilities = get_probabilities(
        current_character
    )


    next_token = random.choices(
        range(vocab_size),
        weights=probabilities,
        k=1
    )[0]


    next_character = itos[next_token]

    generated += next_character

    current_character = next_character


print("\nGENERATED TEXT:\n")
print(generated)