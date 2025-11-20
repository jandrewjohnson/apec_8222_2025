import numpy as np
import time # For QMD date but not strictly needed for NN
import os
import datetime
import shutil
import subprocess

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as patches # For architecture diagram
    # matplotlib.animation is not used in this version for simplicity
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: Matplotlib not found. Visualizations will be generated.")
    # For this script, Matplotlib is essential, so we might exit if not available.

# --- Global Configuration (Ensure these are defined) ---
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
IMAGE_OUTPUT_DIR_NAME = "nn_classification_visualizations"
IMAGE_OUTPUT_DIR = os.path.join(script_dir, IMAGE_OUTPUT_DIR_NAME)
QMD_FILENAME = "nn_classification_report.qmd"


# --- Neural Network Class (Modified to store loss history) ---
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.rand(input_size, hidden_size) * 0.2 - 0.1
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) * 0.2 - 0.1
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        
        self.hidden_layer_input_sum = None # Store net input for visualization
        self.hidden_layer_activation = None
        self.output_layer_input_sum = None # Store net input for visualization
        self.output_layer_activation = None
        
        self.loss_history_ = [] # To store loss during training

        print(f"NN Initialized: Input({input_size}) -> Hidden({hidden_size}) -> Output({output_size})")

    def _sigmoid(self, x):
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def _sigmoid_derivative(self, x_activated):
        return x_activated * (1 - x_activated)

    def forward_propagate(self, X_input, store_sums=False):
        self.hidden_layer_input_sum = np.dot(X_input, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_activation = self._sigmoid(self.hidden_layer_input_sum)

        self.output_layer_input_sum = np.dot(self.hidden_layer_activation, self.weights_hidden_output) + self.bias_output
        self.output_layer_activation = self._sigmoid(self.output_layer_input_sum)
        
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
        self.loss_history_ = [] # Reset loss history
        for epoch in range(epochs):
            predicted_output = self.forward_propagate(X_train)
            self.back_propagate(X_train, y_train, predicted_output, learning_rate)
            
            loss = np.mean(0.5 * (y_train - predicted_output)**2)
            self.loss_history_.append(loss)
            
            if (epoch + 1) % print_loss_every == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        print("--- Training Finished ---")

    def predict_raw_outputs(self, X_input):
        return self.forward_propagate(X_input)

    def predict_class(self, X_input):
        raw_outputs = self.predict_raw_outputs(X_input)
        return np.argmax(raw_outputs, axis=1)

# --- Visualization Functions ---
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_nn_architecture(input_size, hidden_size, output_size, filename="nn_architecture.png"):
    if not MATPLOTLIB_AVAILABLE: return os.path.basename(filename) # Return placeholder
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')

    layer_x_coords = [1, 3, 5]
    layer_labels = ['Input Layer', 'Hidden Layer', 'Output Layer']
    node_counts = [input_size, hidden_size, output_size]
    max_nodes_in_layer = max(node_counts)

    node_radius = 0.25
    y_spacing = 0.8 if max_nodes_in_layer <=5 else 0.6

    for i, (x, count, label) in enumerate(zip(layer_x_coords, node_counts, layer_labels)):
        ax.text(x, max_nodes_in_layer * y_spacing + 1.2 , label, ha='center', fontsize=12)
        # Calculate starting y position to center the nodes
        start_y = (max_nodes_in_layer - count) / 2.0 * y_spacing + node_radius + 0.5
        
        current_layer_nodes = []
        for n in range(count):
            y = start_y + n * y_spacing
            circle = patches.Circle((x, y), node_radius, facecolor='skyblue', edgecolor='black')
            ax.add_patch(circle)
            ax.text(x,y, f"N{n+1}", ha='center',va='center', fontsize=7 if count <=5 else 6)
            current_layer_nodes.append((x,y))
        
        if i > 0: # Add connections from previous layer
            for prev_x, prev_y in prev_layer_nodes:
                for curr_x, curr_y in current_layer_nodes:
                    ax.plot([prev_x + node_radius, curr_x - node_radius], [prev_y, curr_y], 'gray', alpha=0.5, lw=0.8)
        prev_layer_nodes = current_layer_nodes

    ax.set_ylim(0, max_nodes_in_layer * y_spacing + 2)
    ax.set_xlim(0, layer_x_coords[-1] + 1)
    plt.title("Neural Network Architecture", fontsize=14)
    
    filepath = os.path.join(IMAGE_OUTPUT_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=100)
    plt.close(fig)
    return os.path.basename(filename)


def plot_forward_pass_sample(nn_model, sample_x, sample_y_true_onehot, class_names, filename="forward_pass_sample.png"):
    if not MATPLOTLIB_AVAILABLE: return os.path.basename(filename)

    # Perform a forward pass for this sample, storing intermediate sums
    _ = nn_model.forward_propagate(sample_x.reshape(1, -1), store_sums=True) # Reshape to (1, num_features)

    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(3, 3, width_ratios=[1.5, 0.5, 1.5], height_ratios=[1,0.2,1], wspace=0.5, hspace=0.6)

    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.barh(np.arange(len(sample_x)), sample_x, color='lightcoral', height=0.5)
    ax_input.set_yticks(np.arange(len(sample_x)))
    ax_input.set_yticklabels([f'Input {i+1}' for i in range(len(sample_x))])
    ax_input.set_title(f'1. Input Sample:\n{np.round(sample_x,2)}', fontsize=10)
    ax_input.invert_yaxis()
    ax_input.set_xlim(min(0, sample_x.min()-0.1), max(1, sample_x.max()+0.1))


    ax_w_ih = fig.add_subplot(gs[0,2]) # Weights input to hidden
    im_w_ih = ax_w_ih.imshow(nn_model.weights_input_hidden, cmap='viridis', aspect='auto')
    ax_w_ih.set_title(f'2. Weights W_ih ({nn_model.input_size}x{nn_model.hidden_size})\n+ Bias b_h ({nn_model.hidden_size})', fontsize=10)
    fig.colorbar(im_w_ih, ax=ax_w_ih, fraction=0.046, pad=0.04)

    # Hidden Layer Activations
    ax_hidden_act = fig.add_subplot(gs[2,0])
    hidden_act_values = nn_model.hidden_layer_activation.flatten()
    ax_hidden_act.barh(np.arange(len(hidden_act_values)), hidden_act_values, color='mediumseagreen', height=0.5)
    ax_hidden_act.set_yticks(np.arange(len(hidden_act_values)))
    ax_hidden_act.set_yticklabels([f'Hidden {i+1}' for i in range(len(hidden_act_values))])
    ax_hidden_act.set_title(f'3. Hidden Activations:\n{np.round(hidden_act_values,2)}', fontsize=10)
    ax_hidden_act.set_xlim(0,1.1)
    ax_hidden_act.invert_yaxis()

    ax_w_ho = fig.add_subplot(gs[2,2]) # Weights hidden to output
    im_w_ho = ax_w_ho.imshow(nn_model.weights_hidden_output, cmap='viridis', aspect='auto')
    ax_w_ho.set_title(f'4. Weights W_ho ({nn_model.hidden_size}x{nn_model.output_size})\n+ Bias b_o ({nn_model.output_size})', fontsize=10)
    fig.colorbar(im_w_ho, ax=ax_w_ho, fraction=0.046, pad=0.04)

    # Output Layer Activations
    ax_output_act = fig.add_subplot(gs[1,1]) # Center small plot for output
    output_act_values = nn_model.output_layer_activation.flatten()
    predicted_class_idx = np.argmax(output_act_values)
    true_class_idx = np.argmax(sample_y_true_onehot)

    colors_output = ['lightskyblue'] * len(output_act_values)
    colors_output[predicted_class_idx] = 'dodgerblue' # Highlight predicted

    ax_output_act.barh(np.arange(len(output_act_values)), output_act_values, color=colors_output, height=0.4)
    ax_output_act.set_yticks(np.arange(len(output_act_values)))
    ax_output_act.set_yticklabels([f'{class_names[i]}' for i in range(len(output_act_values))], fontsize=8)
    ax_output_act.set_title(f'5. Output Activations\n(Pred: {class_names[predicted_class_idx]}, True: {class_names[true_class_idx]})', fontsize=10)
    ax_output_act.set_xlim(0,1.1)
    ax_output_act.invert_yaxis()
    
    # Conceptual Arrows (simple lines)
    # Input -> Hidden Act
    arrow_y_mid_input_hidden = (ax_input.get_ylim()[0] + ax_hidden_act.get_ylim()[1]) / 2 # approx mid y
    ax_input.annotate("", xy=(ax_input.get_xlim()[1], arrow_y_mid_input_hidden), xycoords='data',
                       xytext=(ax_hidden_act.get_xlim()[0], arrow_y_mid_input_hidden), textcoords='data',
                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='gray', lw=1.5))
    ax_input.text( (ax_input.get_xlim()[1] + ax_hidden_act.get_xlim()[0])/2 , arrow_y_mid_input_hidden + 0.5, 
                   "X @ W_ih + b_h\n-> Sigmoid", ha='center', va='center', fontsize=8, color='gray',
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

    # Hidden Act -> Output Act
    arrow_x_mid_hidden_output = (ax_hidden_act.get_xlim()[1] + ax_output_act.get_xlim()[0]) / 2
    ax_hidden_act.annotate("", xy=(arrow_x_mid_hidden_output, ax_output_act.get_ylim()[1] + 0.1 ), xycoords='data', # (output_act_values.shape[0]-1)/2
                       xytext=(arrow_x_mid_hidden_output, ax_hidden_act.get_ylim()[0] - 0.1), textcoords='data',
                       arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='gray',lw=1.5))
    ax_hidden_act.text(arrow_x_mid_hidden_output, (ax_output_act.get_ylim()[1] + ax_hidden_act.get_ylim()[0])/2,
                       "HiddenAct @ W_ho + b_o\n-> Sigmoid", ha='center', va='center_baseline', fontsize=8, color='gray',
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7), rotation=90)


    plt.suptitle(f"Forward Pass for One Sample (True Class: {class_names[true_class_idx]})", fontsize=14)
    # plt.tight_layout(rect=[0,0,1,0.95]) # tight_layout can conflict with suptitle and complex gridspec
    
    filepath = os.path.join(IMAGE_OUTPUT_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=120)
    plt.close(fig)
    return os.path.basename(filename)


def plot_loss_curve(loss_history, filename="loss_curve.png"):
    if not MATPLOTLIB_AVAILABLE: return os.path.basename(filename)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_history, label="Training Loss (MSE)", color="purple")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error Loss")
    ax.set_title("Training Loss Over Epochs")
    ax.legend()
    ax.grid(True, linestyle=':')
    plt.ylim(bottom=0, top=max(0.1, min(1.0, max(loss_history) * 1.1)) if loss_history else 0.1) # Dynamic Y limit

    filepath = os.path.join(IMAGE_OUTPUT_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=100)
    plt.close(fig)
    return os.path.basename(filename)


def plot_decision_boundaries(nn_model, X_data, y_true_onehot, class_names, filename="decision_boundaries.png"):
    if not MATPLOTLIB_AVAILABLE: return os.path.basename(filename)
    
    y_true_indices = np.argmax(y_true_onehot, axis=1)

    # Create a mesh to plot decision boundaries
    h = .02  # step size in the mesh
    x_min, x_max = X_data[:, 0].min() - 0.2, X_data[:, 0].max() + 0.2
    y_min, y_max = X_data[:, 1].min() - 0.2, X_data[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict classes for all points on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn_model.predict_class(mesh_points)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot decision regions
    # Use a light colormap for regions
    cmap_light = plt.cm.get_cmap(name='Pastel1', lut=len(class_names))
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

    # Plot training points
    # Use a darker colormap for points
    cmap_dark = plt.cm.get_cmap(name='Set1', lut=len(class_names))
    scatter = ax.scatter(X_data[:, 0], X_data[:, 1], c=y_true_indices, cmap=cmap_dark,
                       edgecolor='k', s=50, label=[class_names[i] for i in sorted(np.unique(y_true_indices))])
    
    # Create a legend
    handles, _ = scatter.legend_elements()
    legend_labels = [class_names[i] for i in sorted(np.unique(y_true_indices))]
    ax.legend(handles, legend_labels, title="True Classes")

    ax.set_xlabel("Feature 1 (e.g., Brightness)")
    ax.set_ylabel("Feature 2 (e.g., Size)")
    ax.set_title("Decision Boundaries and Data Points")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    filepath = os.path.join(IMAGE_OUTPUT_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=100)
    plt.close(fig)
    return os.path.basename(filename)
# Ensure these are defined globally or passed appropriately if not already
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
# IMAGE_OUTPUT_DIR_NAME = "nn_classification_visualizations"
# QMD_FILENAME = "nn_classification_report.qmd"

# --- QMD Report Generation (Updated with more detailed explanations for each PNG) ---
def generate_qmd_report(
    nn_params, class_names, X_sample_for_fwd_pass, y_sample_for_fwd_pass_onehot,
    arch_fn, fwd_pass_fn, loss_curve_fn, decision_bnd_fn,
    final_loss, accuracy,
    example_predictions # List of dicts: {'input': X, 'true': T, 'pred': P, 'raw_output': RO}
):
    qmd_content = f"""---
title: "Simple Neural Network Classification Report"
author: "Automated via Python Script"
date: "{datetime.date.today().isoformat()}"
format:
  html:
    toc: true
    code-fold: true
    self-contained: true 
jupyter: python3
---

## 1. Introduction
This report details a simple feedforward neural network, implemented using NumPy, designed for a multi-class classification task. The network aims to classify objects based on two input features into one of three predefined categories. This document will walk through the network's architecture, the training process including forward and backward propagation, and the final classification results.

## 2. Dataset
The dataset consists of samples, each characterized by two numerical features (e.g., "brightness" and "size," typically scaled between 0 and 1). The goal is to classify each sample into one of the following three distinct categories:
"""
    for i, name in enumerate(class_names):
        qmd_content += f"- **Class {i}:** {name}\n"
    
    true_class_idx_sample = np.argmax(y_sample_for_fwd_pass_onehot)
    qmd_content += f"""
As an example, a single data sample might have input features like:
"""
    for i, name in enumerate(class_names):
        qmd_content += f"- **Class {i}:** {name}\n"
    qmd_content += f"""
    A small sample of the input data looks like:

    """

    {str(X_sample_for_fwd_pass)}

    qmd_content +=     f"""

## 3. Neural Network Architecture
The neural network employed has a feedforward architecture with one hidden layer. The specifics are:
- **Input Layer:** Comprises {nn_params['input_size']} neurons, directly corresponding to the {nn_params['input_size']} input features of each data sample.
- **Hidden Layer:** Contains {nn_params['hidden_size']} neurons. This layer allows the network to learn complex, non-linear relationships between the inputs and outputs.
- **Output Layer:** Consists of {nn_params['output_size']} neurons, with each neuron corresponding to one of the {nn_params['output_size']} possible output classes.
The Sigmoid activation function (`1 / (1 + e^-x)`) is used for neurons in both the hidden and output layers. This function squashes the neuron's weighted input sum into a value between 0 and 1.

The following diagram provides a visual representation of this architecture:

![Conceptual Diagram of the Neural Network Architecture]({IMAGE_OUTPUT_DIR_NAME}/{arch_fn})
*Figure 1: Network Architecture. This diagram shows the input layer receiving the features, a fully connected hidden layer, and a fully connected output layer. Each circle represents a neuron, and lines represent weighted connections.*

## 4. Training Process

The network is trained using the backpropagation algorithm with gradient descent to minimize the Mean Squared Error (MSE) loss function.

### 4.1. Forward Propagation
During the forward pass, input data flows through the network from the input layer to the output layer:
1.  **Input to Hidden Layer:** The input features are multiplied by the weights connecting the input layer to the hidden layer (`W_ih`). The corresponding bias (`b_h`) is added to this weighted sum.
2.  **Hidden Layer Activation:** The result from step 1 is passed through the Sigmoid activation function for each hidden neuron. These activations become the input for the next layer.
3.  **Hidden to Output Layer:** The hidden layer activations are multiplied by the weights connecting the hidden layer to the output layer (`W_ho`). The output layer bias (`b_o`) is added.
4.  **Output Layer Activation:** This sum is passed through the Sigmoid activation function for each output neuron. The resulting values (between 0 and 1) are the network's raw predictions or confidence scores for each class. The class corresponding to the output neuron with the highest activation is typically chosen as the predicted class.

The diagram below illustrates the data flow and transformations for a single input sample during a forward pass:

![Illustration of a Forward Pass for a Single Sample]({IMAGE_OUTPUT_DIR_NAME}/{fwd_pass_fn})
*Figure 2: Forward Propagation for One Sample. This figure breaks down the steps: 1. The input sample's feature values. 2. Conceptual representation of the first set of weights (W_ih) and biases. 3. The resulting activations of the hidden layer neurons after applying the sigmoid function. 4. Conceptual representation of the second set of weights (W_ho) and biases. 5. The final activations of the output layer neurons, from which the predicted class is determined.*

### 4.2. Loss Function
The Mean Squared Error (MSE) is used to quantify the difference between the network's predicted outputs (after sigmoid) and the true one-hot encoded labels. For N samples, it's calculated as:
`Loss = (1/N) * Σ_samples (0.5 * Σ_outputs (y_true_output - y_predicted_output)^2)`
The `0.5` is a convention that simplifies the derivative during backpropagation.

### 4.3. Backpropagation
After the forward pass, the error (loss) is calculated. The backpropagation algorithm then computes the gradient of this loss function with respect to each weight and bias in the network. This is done by propagating the error signal backward from the output layer to the input layer, using the chain rule of calculus.
The weights and biases are then updated in the direction opposite to their respective gradients, scaled by a `learning_rate`. This iterative process of forward pass, loss calculation, backpropagation, and weight update aims to progressively minimize the loss and improve the network's accuracy.

The learning progress is typically monitored by plotting the loss value at each epoch (or at regular intervals):

![Training Loss Curve Over Epochs]({IMAGE_OUTPUT_DIR_NAME}/{loss_curve_fn})
*Figure 3: Training Loss Curve. This plot shows the Mean Squared Error (MSE) on the y-axis versus the training epoch on the x-axis. A decreasing trend indicates that the network is learning and its predictions are getting closer to the true labels.*

## 5. Results
After training for {nn_params.get('epochs', 'N/A')} epochs with a learning rate of {nn_params.get('learning_rate', 'N/A')}:
- **Final Training Loss (MSE):** {final_loss:.6f}
- **Accuracy on Training Data:** {accuracy:.2f}%

### 5.1. Classification Visualization and Decision Boundaries
To understand how the trained network separates the different classes in the feature space, we can visualize its decision boundaries. The feature space (defined by Feature 1 and Feature 2) is divided into regions, where each region corresponds to a class predicted by the network.

![Decision Boundaries Learned by the Network]({IMAGE_OUTPUT_DIR_NAME}/{decision_bnd_fn})
*Figure 4: Decision Boundaries and Data Points. The colored regions represent the areas in the feature space where the neural network would predict a specific class. The scattered points are the actual training data samples, colored according to their true class labels. This plot helps visualize how well the network has learned to separate the different categories.*

### 5.2. Example Predictions on Training Data
Here are predictions for the first few samples from the training set:
"""
    for ex_idx, ex in enumerate(example_predictions):
        qmd_content += f"""
 
Sample {ex_idx + 1}
- **Input Features:** `{np.round(ex['input'], 2)}`
- **True Class:** {ex['true_name']} (Internal Index {ex['true_idx']})
- **Predicted Class:** {ex['pred_name']} (Internal Index {ex['pred_idx']})
- **Raw Output Scores (Sigmoid Activations for each class):** `{np.round(ex['raw_output'], 3)}`
- **Assessment:** {'**Correct**' if ex['true_idx'] == ex['pred_idx'] else '**INCORRECT**'}
"""
    qmd_content += """
 
## 6. Conclusion
This simple feedforward neural network, trained using backpropagation and gradient descent, demonstrates its capability to learn non-linear patterns and classify the provided multi-class dataset. The visualizations of the training process (loss curve) and the resulting decision boundaries provide insights into its learning behavior and classification performance. Further improvements could potentially be achieved by tuning hyperparameters such as the number of hidden neurons, learning rate, number of epochs, or by exploring different network architectures or activation functions.
"""
    qmd_file_path = os.path.join(script_dir, QMD_FILENAME)
    try:
        with open(qmd_file_path, 'w', encoding='utf-8') as f: 
            f.write(qmd_content)
        print(f"\nQuarto Markdown report generated: {qmd_file_path}")
        return qmd_file_path
    except Exception as e:
        print(f"Error writing QMD file: {e}")
        return None

# --- Main Script ---
if __name__ == "__main__":
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is essential for this script. Please install it.")
        exit()
    ensure_dir(IMAGE_OUTPUT_DIR)
    
    print("--- Simple Neural Network for Object Classification ---")

    X_data = np.array([[0.1,0.2],[0.2,0.1],[0.8,0.3],[0.7,0.25],[0.3,0.8],[0.25,0.7],[0.15,0.9],[0.9,0.85],[0.05,0.1],[0.95,0.15]])
    y_data_onehot = np.array([[1,0,0],[1,0,0],[0,1,0],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[0,0,1]])
    class_names = ["Small & Dim", "Bright OR Large", "Small & Bright"]
    y_data_indices = np.argmax(y_data_onehot, axis=1)

    nn_params = {
        'input_size': X_data.shape[1], 
        'hidden_size': 5, 
        'output_size': y_data_onehot.shape[1]
    }
    learning_rate = 0.1
    num_epochs = 20000 

    nn_classifier = SimpleNeuralNetwork(
        nn_params['input_size'], nn_params['hidden_size'], nn_params['output_size'], random_seed=45
    )
    
    # Plot architecture before training
    architecture_filename = plot_nn_architecture(
        nn_params['input_size'], nn_params['hidden_size'], nn_params['output_size']
    )

    # Train
    nn_classifier.train(X_data, y_data_onehot, epochs=num_epochs, learning_rate=learning_rate, print_loss_every=num_epochs // 10)

    # Generate visualizations after training
    # Forward pass for one sample
    sample_idx_for_fwd_pass = 2 # e.g., third sample [0.8, 0.3]
    fwd_pass_filename = plot_forward_pass_sample(
        nn_classifier, X_data[sample_idx_for_fwd_pass], y_data_onehot[sample_idx_for_fwd_pass], class_names
    )

    loss_curve_filename = plot_loss_curve(nn_classifier.loss_history_)
    
    decision_boundaries_filename = plot_decision_boundaries(
        nn_classifier, X_data, y_data_onehot, class_names
    )

    # Get final predictions and accuracy
    final_loss = nn_classifier.loss_history_[-1] if nn_classifier.loss_history_ else float('inf')
    predicted_classes_indices = nn_classifier.predict_class(X_data)
    raw_preds_final = nn_classifier.predict_raw_outputs(X_data) # For QMD examples

    correct_count = np.sum(predicted_classes_indices == y_data_indices)
    accuracy = (correct_count / len(X_data)) * 100
    print(f"\nFinal Accuracy on Training Data: {accuracy:.2f}%")

    # Prepare example predictions for QMD
    example_preds_for_qmd = []
    num_examples_to_show = min(5, len(X_data)) # Show first few
    for i in range(num_examples_to_show):
        example_preds_for_qmd.append({
            'input': X_data[i],
            'true_idx': y_data_indices[i],
            'true_name': class_names[y_data_indices[i]],
            'pred_idx': predicted_classes_indices[i],
            'pred_name': class_names[predicted_classes_indices[i]],
            'raw_output': raw_preds_final[i]
        })
    
    # Generate QMD Report
    qmd_file_path = generate_qmd_report(
        nn_params, class_names, X_data[sample_idx_for_fwd_pass], y_data_onehot[sample_idx_for_fwd_pass],
        architecture_filename, fwd_pass_filename, loss_curve_filename, decision_boundaries_filename,
        final_loss, accuracy,
        example_preds_for_qmd
    )

    # Automatically Render the Quarto Document
    if qmd_file_path and os.path.exists(qmd_file_path):
        quarto_executable = shutil.which("quarto")
        if quarto_executable:
            print(f"\nAttempting to render Quarto document: {qmd_file_path}")
            try:
                command = [quarto_executable, "render", qmd_file_path]
                result = subprocess.run(command, capture_output=True, text=True, check=False, cwd=script_dir, timeout=120) # Added timeout
                if result.returncode == 0:
                    output_html_name = os.path.splitext(os.path.basename(qmd_file_path))[0] + ".html"
                    print(f"Quarto document rendered successfully to {os.path.join(script_dir, output_html_name)}")
                else:
                    print("Error during Quarto rendering:"); print("Stdout:", result.stdout); print("Stderr:", result.stderr)
            except subprocess.TimeoutExpired:
                print("Quarto rendering timed out.")
            except Exception as e:
                print(f"An error occurred while trying to render the Quarto document: {e}")
        else:
            print("Quarto executable not found. Please render manually: quarto render " + os.path.basename(qmd_file_path))
    
    print(f"\nCheck '{IMAGE_OUTPUT_DIR}' for PNGs and '{script_dir}' for the QMD/HTML report.")

