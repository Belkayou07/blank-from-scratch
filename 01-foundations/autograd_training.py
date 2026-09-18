import random


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None


    # -------------------------
    # ADDITION
    # -------------------------
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


    # -------------------------
    # MULTIPLICATION
    # -------------------------
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


    # -------------------------
    # NEGATION
    # -------------------------
    def __neg__(self):
        return self * -1


    # -------------------------
    # SUBTRACTION
    # -------------------------
    def __sub__(self, other):
        return self + (-other)


    # -------------------------
    # POWER
    # -------------------------
    def __pow__(self, exponent):
        out = Value(
            self.data ** exponent,
            (self,),
            f"**{exponent}"
        )

        def _backward():
            self.grad += (
                exponent
                * (self.data ** (exponent - 1))
                * out.grad
            )

        out._backward = _backward

        return out


    # -------------------------
    # BACKPROPAGATION
    # -------------------------
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


    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"



# ==================================================
# TRAINING DATA
# ==================================================

data = [
    (1.0, 3.0),
    (2.0, 5.0),
    (3.0, 7.0),
    (4.0, 9.0),
    (5.0, 11.0),
]


# ==================================================
# BLANK PARAMETERS
# ==================================================

w = Value(random.uniform(-1, 1))
b = Value(random.uniform(-1, 1))

learning_rate = 0.01


print("BEFORE TRAINING")
print("w =", w.data)
print("b =", b.data)
print("prediction x=10 =", (w.data * 10) + b.data)
print()



# ==================================================
# TRAINING LOOP
# ==================================================

for epoch in range(1000):

    losses = []


    # -------------------------
    # FORWARD PASS
    # -------------------------

    for x, y in data:

        prediction = w * x + b

        error = prediction - y

        loss = error ** 2

        losses.append(loss)


    # Add all losses together.
    total_loss = sum(losses)

    # Average loss.
    loss = total_loss * (1 / len(data))


    # -------------------------
    # RESET OLD GRADIENTS
    # -------------------------

    w.grad = 0.0
    b.grad = 0.0


    # -------------------------
    # AUTOMATIC BACKPROP
    # -------------------------

    loss.backward()


    # -------------------------
    # GRADIENT DESCENT
    # -------------------------

    w.data -= learning_rate * w.grad
    b.data -= learning_rate * b.grad


    if epoch % 100 == 0:

        print(
            f"epoch={epoch:4d} | "
            f"loss={loss.data:.6f} | "
            f"w={w.data:.4f} | "
            f"b={b.data:.4f} | "
            f"dw={w.grad:.4f} | "
            f"db={b.grad:.4f}"
        )



print()

print("AFTER TRAINING")
print(f"w = {w.data:.4f}")
print(f"b = {b.data:.4f}")

prediction = w.data * 10 + b.data

print(f"Prediction for x=10: {prediction:.4f}")
print("Correct answer:      21.0000")