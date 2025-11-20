import numpy as np

# --- Neural Network Class (largely the same, output interpretation will change) ---
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)

        self.weights_input_hidden = np.random.rand(input_size, hidden_size) * 0.2 - 0.1 # Slightly wider init
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) * 0.2 - 0.1
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        
        self.hidden_layer_activation = None
        self.output_layer_activation = None
        print(f"NN Initialized: Input({input_size}) -> Hidden({hidden_size}) -> Output({output_size})")

    def _sigmoid(self, x):
        # Clip x to avoid overflow/underflow in exp
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def _sigmoid_derivative(self, x_activated): # x_activated is sigmoid(z)
        return x_activated * (1 - x_activated)

    def forward_propagate(self, X_input):
        hidden_layer_weighted_sum = np.dot(X_input, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_activation = self._sigmoid(hidden_layer_weighted_sum)

        output_layer_weighted_sum = np.dot(self.hidden_layer_activation, self.weights_hidden_output) + self.bias_output
        self.output_layer_activation = self._sigmoid(output_layer_weighted_sum) # Sigmoid for each output neuron
        
        return self.output_layer_activation

    def back_propagate(self, X_input, y_true, predicted_output, learning_rate):
        output_error = y_true - predicted_output
        delta_output = output_error * self._sigmoid_derivative(predicted_output)

        hidden_layer_error = np.dot(delta_output, self.weights_hidden_output.T)
        delta_hidden = hidden_layer_error * self._sigmoid_derivative(self.hidden_layer_activation)
        
        grad_weights_hidden_output = np.dot(self.hidden_layer_activation.T, delta_output)
        self.weights_hidden_output += learning_rate * grad_weights_hidden_output
        
        grad_bias_output = np.sum(delta_output, axis=0, keepdims=True)
        self.bias_output += learning_rate * grad_bias_output

        grad_weights_input_hidden = np.dot(X_input.T, delta_hidden)
        self.weights_input_hidden += learning_rate * grad_weights_input_hidden

        grad_bias_hidden = np.sum(delta_hidden, axis=0, keepdims=True)
        self.bias_hidden += learning_rate * grad_bias_hidden

    def train(self, X_train, y_train, epochs, learning_rate, print_loss_every=100):
        print("\n--- Training Started ---")
        for epoch in range(epochs):
            predicted_output = self.forward_propagate(X_train)
            self.back_propagate(X_train, y_train, predicted_output, learning_rate)
            
            if (epoch + 1) % print_loss_every == 0 or epoch == 0:
                loss = np.mean(0.5 * (y_train - predicted_output)**2)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        print("--- Training Finished ---")

    def predict_raw_outputs(self, X_input):
        return self.forward_propagate(X_input)

    def predict_class(self, X_input):
        """Predicts the class label (index of the neuron with max activation)."""
        raw_outputs = self.predict_raw_outputs(X_input)
        return np.argmax(raw_outputs, axis=1) # Get the index of the max value along rows

# --- Example Usage: Simple Object Classification ---
if __name__ == "__main__":
    print("--- Simple Neural Network for Object Classification ---")

    # Features: [brightness (0-1), size (0-1)]
    # Lower values = dimmer/smaller, Higher values = brighter/larger
    X_data = np.array([
        [0.1, 0.2], # Small & Dim
        [0.2, 0.1], # Small & Dim
        [0.8, 0.3], # Bright, Med-Small Size
        [0.7, 0.25],# Bright, Small Size
        [0.3, 0.8], # Med-Dim, Large Size
        [0.25, 0.7],# Dim, Large Size
        [0.15, 0.9],# Dim, Very Large
        [0.9, 0.85],# Bright & Large (should lean towards class 1)
        [0.05, 0.1],# Very Small & Dim
        [0.95, 0.15]# Very Bright & Small
    ])

    # Target Classes (one-hot encoded):
    # Class 0: "Small & Dim" (e.g., brightness < 0.3 AND size < 0.3)
    # Class 1: "Bright OR Large" (e.g., brightness > 0.6 OR size > 0.6)
    # Class 2: "Small & Bright" (e.g., brightness > 0.6 AND size < 0.4) - more specific
    
    # Corresponding y_data (one-hot encoded)
    # [Class0, Class1, Class2]
    y_data = np.array([
        [1, 0, 0], # [0.1, 0.2] -> Small & Dim
        [1, 0, 0], # [0.2, 0.1] -> Small & Dim
        [0, 1, 0], # [0.8, 0.3] -> Bright (fits Bright OR Large)
        [0, 0, 1], # [0.7, 0.25]-> Small & Bright
        [0, 1, 0], # [0.3, 0.8] -> Large (fits Bright OR Large)
        [0, 1, 0], # [0.25, 0.7]-> Large
        [0, 1, 0], # [0.15, 0.9]-> Very Large
        [0, 1, 0], # [0.9, 0.85]-> Bright & Large (fits Bright OR Large best)
        [1, 0, 0], # [0.05, 0.1]-> Small & Dim
        [0, 0, 1]  # [0.95, 0.15]-> Small & Bright
    ])
    class_names = ["Small & Dim", "Bright OR Large", "Small & Bright"]

    # Network parameters
    input_layer_size = X_data.shape[1] 
    hidden_layer_size = 5             # Can be tuned
    output_layer_size = y_data.shape[1] # Should be 3 for our 3 classes
    
    learning_rate = 0.1
    num_epochs = 20000 # Might need more for multi-class

    # Create and train the neural network
    nn_classifier = SimpleNeuralNetwork(input_layer_size, hidden_layer_size, output_layer_size, random_seed=45)
    nn_classifier.train(X_data, y_data, epochs=num_epochs, learning_rate=learning_rate, print_loss_every=num_epochs // 20)

    # Test the trained network
    print("\n--- Testing Trained Network ---")
    raw_predictions = nn_classifier.predict_raw_outputs(X_data)
    predicted_classes = nn_classifier.predict_class(X_data)
    true_classes = np.argmax(y_data, axis=1) # Convert one-hot y_data back to class indices for comparison

    print("\nInput Data (X_data):")
    for i in range(len(X_data)):
        print(f"  Sample {i+1}: {X_data[i]}")

    print("\nTrue Outputs (One-Hot Encoded y_data):")
    for i in range(len(y_data)):
        print(f"  Sample {i+1}: {y_data[i]} (Class: {class_names[true_classes[i]]})")
    
    print("\nPredicted Raw Outputs (Sigmoid Activations per Class):")
    for i in range(len(raw_predictions)):
        print(f"  Sample {i+1}: {np.round(raw_predictions[i], 3)}")

    print("\nPredicted Class Index & Name:")
    correct_count = 0
    for i in range(len(predicted_classes)):
        is_correct = "Correct" if predicted_classes[i] == true_classes[i] else "INCORRECT"
        if predicted_classes[i] == true_classes[i]:
            correct_count +=1
        print(f"  Sample {i+1}: Input {X_data[i]} -> Predicted Class: {predicted_classes[i]} ({class_names[predicted_classes[i]]}) "
              f"| True Class: {true_classes[i]} ({class_names[true_classes[i]]}) -> {is_correct}")

    accuracy = (correct_count / len(X_data)) * 100
    print(f"\nAccuracy on this dataset: {accuracy:.2f}%")

    if accuracy > 90: # Arbitrary threshold for "good enough" on this small set
        print("The neural network learned to classify the objects reasonably well!")
    else:
        print("The network performance can be improved. Try adjusting hidden layer size, epochs, or learning rate.")

    # Example of predicting a new, unseen sample
    print("\n--- Predicting a New Sample ---")
    new_sample_bright_large = np.array([[0.8, 0.7]]) # Expected: Bright OR Large (Class 1)
    pred_raw_new = nn_classifier.predict_raw_outputs(new_sample_bright_large)
    pred_class_new = nn_classifier.predict_class(new_sample_bright_large)
    print(f"New sample {new_sample_bright_large[0]}:")
    print(f"  Raw Outputs: {np.round(pred_raw_new[0], 3)}")
    print(f"  Predicted Class: {pred_class_new[0]} ({class_names[pred_class_new[0]]})")

    new_sample_small_dim = np.array([[0.1, 0.1]]) # Expected: Small & Dim (Class 0)
    pred_raw_new_2 = nn_classifier.predict_raw_outputs(new_sample_small_dim)
    pred_class_new_2 = nn_classifier.predict_class(new_sample_small_dim)
    print(f"New sample {new_sample_small_dim[0]}:")
    print(f"  Raw Outputs: {np.round(pred_raw_new_2[0], 3)}")
    print(f"  Predicted Class: {pred_class_new_2[0]} ({class_names[pred_class_new_2[0]]})")