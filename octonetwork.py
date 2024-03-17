class OctonionNN:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.activations = []

        # Initialize weights and biases for each layer
        for i in range(1, len(layers)):
            self.weights.append([[random_octonion() for _ in range(layers[i-1])] for _ in range(layers[i])])
            self.biases.append([random_octonion() for _ in range(layers[i])])

    def octonion_norm(self, octonion):
        return np.sqrt(octonion.real ** 2 + octonion.e1 ** 2 + octonion.e2 ** 2 + octonion.e3 ** 2 +
                       octonion.e4 ** 2 + octonion.e5 ** 2 + octonion.e6 ** 2 + octonion.e7 ** 2)

    def normalize_octonion(self, octonion):
        norm = self.octonion_norm(octonion)
        return Octonion(octonion.real / norm, octonion.e1 / norm, octonion.e2 / norm,
                        octonion.e3 / norm, octonion.e4 / norm, octonion.e5 / norm,
                        octonion.e6 / norm, octonion.e7 / norm)

    def octonion_sigmoid(self, z):
        return 1 / (1 + np.exp(-self.octonion_norm(z)))

    def forward_propagate(self, x):
        self.activations = [x]
        for i in range(len(self.layers) - 1):
            z = np.array([random_octonion() for _ in range(self.layers[i+1])])
            for j in range(self.layers[i+1]):
                for k in range(self.layers[i]):
                    z[j] += self.weights[i][j][k] * self.activations[i][k]
                z[j] += self.biases[i][j]
            a = np.array([self.normalize_octonion(val) for val in z])
            self.activations.append(a)
        return self.activations[-1]

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                x = np.array(x, ndmin=2).T
                y = np.array(y, ndmin=2).T

                # Forward propagation
                output = self.forward_propagate(x)

                # Backward propagation (octonion-valued gradients)
                deltas = [output - y]
                for i in range(len(self.layers) - 2, 0, -1):
                    delta = np.array
