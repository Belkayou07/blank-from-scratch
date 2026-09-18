import random


# --------------------------------------------------
# TRAINING DATA
# --------------------------------------------------
# The hidden rule is:
# y = 2x + 1
#
# The model is NOT told this rule.
# It only sees examples.

data = [
    (1.0, 3.0),
    (2.0, 5.0),
    (3.0, 7.0),
    (4.0, 9.0),
    (5.0, 11.0),
]


# --------------------------------------------------
# BLANK MODEL
# --------------------------------------------------
# Our model is:
#
# prediction = w * x + b
#
# w and b are the parameters the model must learn.

w = random.uniform(-1.0, 1.0)
b = random.uniform(-1.0, 1.0)


# Controls how large each learning step is.
learning_rate = 0.01


print("BEFORE TRAINING")
print(f"w = {w:.4f}")
print(f"b = {b:.4f}")
print(f"Prediction for x=10: {w * 10 + b:.4f}")
print()


# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------

for epoch in range(1000):

    total_loss = 0.0

    # Gradients for our two parameters.
    dw = 0.0
    db = 0.0

    for x, y in data:

        # 1. MODEL MAKES A PREDICTION
        prediction = w * x + b


        # 2. MEASURE HOW WRONG IT IS
        error = prediction - y

        loss = error ** 2

        total_loss += loss


        # 3. CALCULATE HOW w AND b CONTRIBUTED
        #    TO THE ERROR
        dw += 2 * error * x
        db += 2 * error


    # Average everything over the dataset.
    n = len(data)

    total_loss /= n
    dw /= n
    db /= n


    # 4. LEARN:
    #    modify the parameters in the direction
    #    that reduces the error.
    w -= learning_rate * dw
    b -= learning_rate * db


    if epoch % 100 == 0:
        print(
            f"epoch={epoch:4d} | "
            f"loss={total_loss:.6f} | "
            f"w={w:.4f} | "
            f"b={b:.4f}"
        )


print()
print("AFTER TRAINING")
print(f"w = {w:.4f}")
print(f"b = {b:.4f}")
print(f"Prediction for x=10: {w * 10 + b:.4f}")
print("Correct answer:      21.0000")