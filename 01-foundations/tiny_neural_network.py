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


    # NEW OPERATION
    def tanh(self):

        t = math.tanh(self.data)

        out = Value(
            t,
            (self,),
            "tanh"
        )

        def _backward():

            # derivative of tanh:
            # 1 - tanh(x)^2

            self.grad += (1 - t ** 2) * out.grad

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
# ONE NEURON
# ==================================================

class Neuron:

    def __init__(self, number_of_inputs):

        # One weight for every input.
        self.w = [
            Value(random.uniform(-1, 1))
            for _ in range(number_of_inputs)
        ]

        # One bias.
        self.b = Value(random.uniform(-1, 1))


    def __call__(self, x):

        # Weighted sum:
        #
        # w1*x1 + w2*x2 + ... + bias

        z = sum(
            (weight * input_value
             for weight, input_value in zip(self.w, x)),
            self.b
        )

        # Nonlinear activation.
        return z.tanh()


    def parameters(self):
        return self.w + [self.b]



# ==================================================
# A LAYER OF NEURONS
# ==================================================

class Layer:

    def __init__(self, number_of_inputs, number_of_neurons):

        self.neurons = [
            Neuron(number_of_inputs)
            for _ in range(number_of_neurons)
        ]


    def __call__(self, x):

        return [
            neuron(x)
            for neuron in self.neurons
        ]


    def parameters(self):

        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]



# ==================================================
# MULTI-LAYER PERCEPTRON
# ==================================================

class MLP:

    def __init__(self, number_of_inputs, layer_sizes):

        sizes = [number_of_inputs] + layer_sizes

        self.layers = [
            Layer(sizes[i], sizes[i + 1])
            for i in range(len(layer_sizes))
        ]


    def __call__(self, x):

        for layer in self.layers:
            x = layer(x)

        # Our final layer contains one neuron.
        return x[0]


    def parameters(self):

        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]



# ==================================================
# BUILD A COMPLETELY RANDOM NETWORK
# ==================================================

random.seed(42)

model = MLP(
    number_of_inputs=2,
    layer_sizes=[4, 1]
)


print("Number of parameters:", len(model.parameters()))



# ==================================================
# XOR TRAINING DATA
# ==================================================

data = [
    ([0.0, 0.0], -1.0),
    ([0.0, 1.0],  1.0),
    ([1.0, 0.0],  1.0),
    ([1.0, 1.0], -1.0),
]



# ==================================================
# BEFORE TRAINING
# ==================================================

print("\nBEFORE TRAINING")

for x, target in data:

    prediction = model(x)

    print(
        x,
        "prediction:",
        round(prediction.data, 4),
        "target:",
        target
    )



# ==================================================
# TRAINING
# ==================================================

learning_rate = 0.05


for epoch in range(2000):

    predictions = [
        model(x)
        for x, target in data
    ]


    # Mean squared error.
    total_loss = sum(
        (prediction - target) ** 2
        for prediction, (_, target)
        in zip(predictions, data)
    )

    loss = total_loss * (1 / len(data))


    # Clear gradients from previous epoch.
    for parameter in model.parameters():
        parameter.grad = 0.0


    # Our autograd engine calculates all
    # 17 parameter gradients.
    loss.backward()


    # Gradient descent.
    for parameter in model.parameters():

        parameter.data -= (
            learning_rate * parameter.grad
        )


    if epoch % 200 == 0:

        print(
            f"epoch={epoch:4d} | "
            f"loss={loss.data:.6f}"
        )



# ==================================================
# AFTER TRAINING
# ==================================================

print("\nAFTER TRAINING")

for x, target in data:

    prediction = model(x)

    print(
        x,
        "prediction:",
        round(prediction.data, 4),
        "target:",
        target
    )