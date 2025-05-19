import numpy as np


class mlp:
    def __init__(self, input=[0, 1], expected_output=[1]):
        self.w2 = np.random.randn(2, 2)
        self.w1 = np.random.randn(1, 2)
        self.b2 = np.random.randn(2, 1)
        self.b1 = np.random.randn(1, 1)
        self.expected_output = expected_output
        self.input = np.atleast_2d(input).T
        self.forward_state = "initialized"
        self.learning_rate = 0.01
        self.loss = None

    def sigmoid(self, z):
        activation = 1 / (1 + np.exp(-z))
        return activation

    def forwardPass(self):
        hidden_layer = self.w2 @ self.input
        self.z2 = hidden_layer + self.b2

        self.a2 = self.sigmoid(self.z2)

        output_neuron = self.w1 @ self.a2
        self.z1 = output_neuron + self.b1

        self.output = self.sigmoid(self.z1)

        if not self.forward_state == "running":
            self.forward_state = "running"

        return self.output

    def MSE(self):
        output = self.forwardPass()
        self.loss = ((output - self.expected_output) ** 2) / 2

        return self.loss

    def gradientDescent(self):
        dCdz1 = (self.output - self.expected_output) * (
            (self.output) * (1 - self.output)
        )
        dCdw1 = dCdz1 * self.a2.T
        dCdb1 = dCdz1 * 1

        dCdz2 = dCdz1 * (self.w1 @ (self.a2 * (1 - self.a2)))
        dCdw2 = dCdz2 * self.input
        dCdb2 = dCdz2 * 1

        self.w1 = self.w1 - self.learning_rate * dCdw1
        self.b1 = self.b1 - self.learning_rate * dCdb1
        self.w2 = self.w2 - self.learning_rate * dCdw2
        self.b2 = self.b2 - self.learning_rate * dCdb2

    def backwardPass(self):
        self.epochs = 0
        loss = self.MSE()
        print(f"Initial loss: {loss}")
        while self.epochs < 15000:
            self.forwardPass()
            self.gradientDescent()
            self.epochs += 1

        loss2 = self.MSE()
        print(f"Final loss: {loss2}")
        print(f"Expected Output: {self.expected_output}")
        print(f"Final Output: {self.output}")


input = np.array([1, 1])
expected = np.array([0])
nn1 = mlp(input, expected)
nn1.backwardPass()
