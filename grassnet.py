# Define the GrassmannNN class
class GrassmannNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights and biases as Grassmann numbers
        self.weights_input_hidden = self.initialize_weights(hidden_dim, input_dim)
        self.weights_hidden_output = self.initialize_weights(output_dim, hidden_dim)
        self.bias_hidden = self.initialize_bias(hidden_dim)
        self.bias_output = self.initialize_bias(output_dim)

    def initialize_weights(self, rows, cols):
        return [[GrassmannNumber(random.uniform(-1, 1)) for _ in range(cols)] for _ in range(rows)]

    def initialize_bias(self, rows):
        return [GrassmannNumber(random.uniform(-1, 1)) for _ in range(rows)]

    def forward_pass(self, x):
        # Convert input to Grassmann numbers
        inputs = [GrassmannNumber(value) for value in x]

        # Hidden layer computation
        hidden_output = []
        for i in range(self.hidden_dim):
            weighted_sum = GrassmannNumber()
            for j in range(self.input_dim):
                weighted_sum += self.weights_input_hidden[i][j] * inputs[j]
            hidden_output.append(weighted_sum + self.bias_hidden[i])

        # Activation function (sigmoid)
        sigmoid = lambda z: GrassmannNumber(1) / (GrassmannNumber(1) + GrassmannNumber(np.exp(-z.scalar)))

        hidden_output = [sigmoid(val) for val in hidden_output]

        # Output layer computation
        output = []
        for i in range(self.output_dim):
            weighted_sum = GrassmannNumber()
            for j in range(self.hidden_dim):
                weighted_sum += self.weights_hidden_output[i][j] * hidden_output[j]
            output.append(weighted_sum + self.bias_output[i])

        return output, hidden_output

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                # Forward pass
                output, hidden_output = self.forward_pass(x)

                # Calculate loss (mean squared error)
                loss = sum([(val.scalar - target) ** 2 for val, target in zip(output, y)]) / self.output_dim

                # Backward pass (gradient descent update)
                d_loss = 2 * (output[0].scalar - y[0]) / self.output_dim  # Loss derivative with respect to output
                d_output = [GrassmannNumber(d_loss) for _ in range(self.output_dim)]

                # Update output layer weights and biases
                for i in range(self.output_dim):
                    for j in range(self.hidden_dim):
                        self.weights_hidden_output[i][j] -= learning_rate * d_output[i] * hidden_output[j]
                    self.bias_output[i] -= learning_rate * d_output[i]

                # Backpropagate to hidden layer
                d_hidden = [GrassmannNumber() for _ in range(self.hidden_dim)]
                for i in range(self.output_dim):
                    for j in range(self.hidden_dim):
                        d_hidden[j] += d_output[i] * self.weights_hidden_output[i][j]

                # Update hidden layer weights and biases
                for i in range(self.hidden_dim):
                    for j in range(self.input_dim):
                        self.weights_input_hidden[i][j] -= learning_rate * d_hidden[i] * x[j]
                    self.bias_hidden[i] -= learning_rate * d_hidden[i]

# Example usage
input_dim = 2
hidden_dim = 3
output_dim = 1

# Create a GrassmannNN instance
nn = GrassmannNN(input_dim, hidden_dim, output_dim)

# Sample dataset (XOR function)
x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [[0], [1], [1], [0]]

# Train the GrassmannNN
nn.train(x_train, y_train, epochs=1000, learning_rate=0.1)

# Test the trained network
for x, y in zip(x_train, y_train):
    output, _ = nn.forward_pass(x)
    print(f"Input: {x}, Target: {y}, Predicted: {output[0].scalar}")
