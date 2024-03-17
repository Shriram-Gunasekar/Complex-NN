class ComplexNN:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.activations = []

        # Initialize weights and biases for each layer
        for i in range(1, len(layers)):
            # Use complex numbers for weights and biases
            w_real = np.random.randn(layers[i], layers[i-1])
            w_imag = np.random.randn(layers[i], layers[i-1])
            b_real = np.random.randn(layers[i], 1)
            b_imag = np.random.randn(layers[i], 1)

            self.weights.append(w_real + 1j * w_imag)
            self.biases.append(b_real + 1j * b_imag)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward_propagate(self, x):
        self.activations = [x]
        for i in range(len(self.layers) - 1):
            z = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            a = self.sigmoid(z)
            self.activations.append(a)
        return self.activations[-1]

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                x = np.array(x, ndmin=2).T
                y = np.array(y, ndmin=2).T

                # Forward propagation
                output = self.forward_propagate(x)

                # Backward propagation (complex-valued gradients)
                deltas = [output - y]
                for i in range(len(self.layers) - 2, 0, -1):
                    delta = np.dot(self.weights[i].T.conj(), deltas[-1]) * \
                            self.activations[i] * (1 - self.activations[i])
                    deltas.append(delta)

                # Update weights and biases
                deltas.reverse()
                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * np.dot(deltas[i], self.activations[i].T.conj())
                    self.biases[i] -= learning_rate * deltas[i]
