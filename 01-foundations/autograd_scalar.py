class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data

        # Gradient:
        # how much the final result changes
        # if this value changes slightly
        self.grad = 0.0

        # Values that created this value
        self._prev = set(_children)

        # Operation that created this value
        self._op = _op

        # Function used during backpropagation
        self._backward = lambda: None


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            (self, other),
            "+"
        )

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out


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


    def backward(self):

        # Build the computation graph in the correct order.
        topo = []
        visited = set()

        def build_topology(value):
            if value not in visited:
                visited.add(value)

                for child in value._prev:
                    build_topology(child)

                topo.append(value)

        build_topology(self)

        # The final result changes 1-for-1 with itself.
        self.grad = 1.0

        # Walk backwards through the graph.
        for value in reversed(topo):
            value._backward()


    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


# --------------------------------------------------
# TEST OUR ENGINE
# --------------------------------------------------

a = Value(2.0)
b = Value(3.0)
c = Value(1.0)

d = a * b
z = d + c

print("FORWARD PASS")
print("a =", a)
print("b =", b)
print("c =", c)
print("d = a * b =", d)
print("z = d + c =", z)

z.backward()

print()
print("AFTER BACKPROPAGATION")
print("a =", a)
print("b =", b)
print("c =", c)
print("d =", d)
print("z =", z)