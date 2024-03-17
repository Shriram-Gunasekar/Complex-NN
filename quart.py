import numpy as np

class Quaternion:
    def __init__(self, real, i, j, k):
        self.real = real
        self.i = i
        self.j = j
        self.k = k

    def __repr__(self):
        return f"({self.real}, {self.i}, {self.j}, {self.k})"

    def __add__(self, other):
        return Quaternion(self.real + other.real, self.i + other.i, self.j + other.j, self.k + other.k)

    def __mul__(self, other):
        real = self.real * other.real - self.i * other.i - self.j * other.j - self.k * other.k
        i = self.real * other.i + self.i * other.real + self.j * other.k - self.k * other.j
        j = self.real * other.j - self.i * other.k + self.j * other.real + self.k * other.i
        k = self.real * other.k + self.i * other.j - self.j * other.i + self.k * other.real
        return Quaternion(real, i, j, k)

class QuaternionNN:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.activations = []

        # Initialize weights and biases for each layer
        for i in range(1, len(layers)):
            w_real = np.random.randn(layers[i], layers[i-1])
            w_i = np.random.randn(layers[i], layers[i-1])
            w_j = np.random.randn(layers[i], layers[i-1])
            w_k = np.random.randn(layers[i], layers[i-1])
            b_real = np.random.randn(layers[i], 1)
            b_i = np.random.randn(layers[i], 1)
            b_j = np.random.randn(layers[i], 1)
            b_k = np.random.randn(layers[i], 1)

            self.weights.append([Quaternion(w_real[j, k], w_i[j, k], w_j[j, k], w_k[j, k]) for j in range(layers[i]) for k in range(layers[i-1])])
            self.biases.append(Quaternion(b_real[j, 0], b_i[j, 0], b_j[j, 0], b_k[j, 0]) for j in range(layers[i]))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward_propagate(self, x):
        self.activations = [x]
        for i in range(len(self.layers) - 1):
            z = np.array([Quaternion(0, 0, 0, 0) for _ in range(self.layers[i+1])])
            for j in range(self.layers[i+1]):
                for k in range(self.layers[i]):
                    z[j] += self.weights[i][j*self.layers[i] + k] * self.activations[i][k]
                z[j] += self.biases[i][j]
            a = np.array([self.sigmoid(val.real) for val in z])
            self.activations.append(a)
        return self.activations[-1]

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                x = np.array(x, ndmin=2).T
                y = np.array(y, ndmin=2).T

                # Forward propagation
                output = self.forward_propagate(x)

                # Backward propagation (quaternion-valued gradients)
                deltas = [output - y]
                for i in range(len(self.layers) - 2, 0, -1):
                    delta = np.array([Quaternion(0, 0, 0, 0) for _ in range(self.layers[i])])
                    for j in range(self.layers[i]):
                        for k in range(self.layers[i+1]):
                            delta[j] += self.weights[i][k*self.layers[i] + j] * deltas[-1][k]
                        delta[j] *= self.activations[i][j] * (1 - self.activations[i][j])
                    deltas.append(delta)

                # Update weights and biases
                deltas.reverse()
                for i in range(len(self.weights)):
                    for j in range(self.layers[i+1]):
                        for k in range(self.layers[i]):
                            self.weights[i][j*self.layers[i] + k] -= learning_rate * deltas[i][k] * self.activations[i][k].real
                        self.biases[i][j] -= learning_rate * deltas[i][j].real
