import numpy as np
import shutil, subprocess
import time
import os
import datetime # For QMD date

# --- Matplotlib for Visualization ---
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: Matplotlib not found. Visualizations will not be generated.")
    print("Please install Matplotlib: pip install matplotlib")

# --- Global Configuration ---
GRID_SIZE = 7 # Dimension of the digit grid (7x7)
N_FEATURES = GRID_SIZE * GRID_SIZE

# --- Perceptron Class (largely unchanged, robust) ---
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=100, random_state=None):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        if random_state:
            np.random.seed(random_state)
        self.weights_ = None
        self.bias_ = 0.0 # Initialize as float
        self.errors_ = []
        self.weight_history_ = []

    def _activation_function(self, x):
        return np.where(x >= 0, 1, 0)

    def _net_input(self, X):
        X_proc = np.atleast_2d(X)
        # This will return a 1-element array if X_proc is a single sample
        return np.dot(X_proc, self.weights_) + self.bias_
    def fit(self, X, y, verbose=False, store_history_interval=None): # Added store_history_interval
        n_samples, n_features_data = X.shape
        if n_features_data != N_FEATURES:
            raise ValueError(f"Input data feature size {n_features_data} does not match expected {N_FEATURES}")

        self.weights_ = np.zeros(n_features_data)
        self.bias_ = 0.0 
        self.errors_ = []
        self.weight_history_ = [] # Clear history for new fit

        # Store initial state
        if store_history_interval is not None:
            self.weight_history_.append((self.weights_.copy(), self.bias_))

        update_counter = 0 # To control history storage based on updates

        for epoch in range(self.n_iters):
            errors_in_epoch = 0
            for i in range(n_samples):
                xi = X[i]; target = y[i]
                prediction_activated = self._activation_function(self._net_input(xi)) 
                update_val = target - prediction_activated
                if isinstance(update_val, np.ndarray): update_val = update_val.item() 
                
                update = self.learning_rate * update_val

                if update != 0:
                    self.weights_ += update * xi
                    self.bias_ += update 
                    errors_in_epoch += 1
                    update_counter += 1

                    # Store weights if interval is met
                    if store_history_interval is not None and update_counter % store_history_interval == 0:
                        self.weight_history_.append((self.weights_.copy(), self.bias_))
            
            self.errors_.append(errors_in_epoch)
            # Optionally store at end of epoch if interval is based on epochs
            # if store_history_interval is not None and store_history_interval == "epoch":
            #    self.weight_history_.append((self.weights_.copy(), self.bias_))

            if verbose and (epoch < 3 or epoch % (self.n_iters // 10 or 1) == 0 or errors_in_epoch == 0):
                 print(f"    Epoch {epoch+1}/{self.n_iters} - Updates: {errors_in_epoch}, Bias: {float(self.bias_):.4f}")
            
            if errors_in_epoch == 0 and epoch > 0:
                if verbose: print(f"    Converged at epoch {epoch+1}!")
                break
        
        # Store final state if not already captured by interval
        if store_history_interval is not None and (not self.weight_history_ or \
            (not np.array_equal(self.weight_history_[-1][0], self.weights_) or self.weight_history_[-1][1] != self.bias_)):
            self.weight_history_.append((self.weights_.copy(), self.bias_))

        if verbose: print(f"  Final bias: {float(self.bias_):.4f}")
        return self

    def predict(self, X):
        # _net_input returns array, _activation_function handles it, flatten ensures 1D output
        return self._activation_function(self._net_input(X)).flatten()


# --- 7x7 Digit Patterns (Unchanged) ---
DIGIT_PATTERNS_7x7 = {
    0: np.array([ [0,0,1,1,1,0,0],[0,1,0,0,0,1,0],[1,0,0,0,0,0,1],[1,0,0,0,0,0,1],[1,0,0,0,0,0,1],[0,1,0,0,0,1,0],[0,0,1,1,1,0,0]]),
    1: np.array([ [0,0,0,1,0,0,0],[0,0,1,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,1,1,1,0,0]]),
    2: np.array([ [0,1,1,1,1,0,0],[1,0,0,0,0,1,0],[0,0,0,0,1,0,0],[0,0,0,1,0,0,0],[0,0,1,0,0,0,0],[0,1,0,0,0,0,0],[1,1,1,1,1,1,0]]),
    3: np.array([ [0,1,1,1,1,0,0],[1,0,0,0,0,1,0],[0,0,0,0,0,1,0],[0,0,1,1,1,0,0],[0,0,0,0,0,1,0],[1,0,0,0,0,1,0],[0,1,1,1,1,0,0]]),
    4: np.array([ [0,0,0,0,1,0,0],[0,0,0,1,1,0,0],[0,0,1,0,1,0,0],[0,1,0,0,1,0,0],[1,1,1,1,1,1,1],[0,0,0,0,1,0,0],[0,0,0,0,1,0,0]]),
    5: np.array([ [0,1,1,1,1,1,0],[0,1,0,0,0,0,0],[0,1,1,1,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,1,0],[1,0,0,0,0,1,0],[0,1,1,1,1,0,0]]),
    6: np.array([ [0,0,0,1,1,0,0],[0,0,1,0,0,0,0],[0,1,0,0,0,0,0],[0,1,1,1,1,0,0],[1,0,0,0,0,1,0],[0,1,0,0,0,1,0],[0,0,1,1,1,0,0]]),
    7: np.array([ [1,1,1,1,1,1,0],[0,0,0,0,0,1,0],[0,0,0,0,1,0,0],[0,0,0,1,0,0,0],[0,0,1,0,0,0,0],[0,0,1,0,0,0,0],[0,0,1,0,0,0,0]]),
    8: np.array([ [0,0,1,1,1,0,0],[0,1,0,0,0,1,0],[0,1,0,0,0,1,0],[0,0,1,1,1,0,0],[0,1,0,0,0,1,0],[0,1,0,0,0,1,0],[0,0,1,1,1,0,0]]),
    9: np.array([ [0,0,1,1,1,0,0],[0,1,0,0,0,1,0],[1,0,0,0,0,1,0],[0,1,1,1,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,1,0,0],[0,0,1,1,0,0,0]])
}

# --- Helper Functions (Unchanged) ---
def flatten_patterns(patterns_dict):
    X_list, y_list = [], []
    for digit, pattern in patterns_dict.items():
        if pattern.shape != (GRID_SIZE, GRID_SIZE):
             raise ValueError(f"Pattern for digit {digit} has incorrect shape {pattern.shape}. Expected ({GRID_SIZE},{GRID_SIZE})")
        X_list.append(pattern.flatten())
        y_list.append(digit)
    return np.array(X_list), np.array(y_list)

def add_noise(X_original, y_original, num_noisy_versions=5, noise_level=0.05):
    X_noisy_list, y_noisy_list = [], []
    n_features = X_original.shape[1]
    for i in range(X_original.shape[0]):
        original_sample, original_label = X_original[i], y_original[i]
        X_noisy_list.append(original_sample); y_noisy_list.append(original_label)
        for _ in range(num_noisy_versions):
            noisy_sample = original_sample.copy()
            num_flips = int(noise_level * n_features)
            if num_flips == 0 and noise_level > 0 : num_flips = 1 
            flip_indices = np.random.choice(n_features, size=num_flips, replace=False)
            for idx in flip_indices: noisy_sample[idx] = 1 - noisy_sample[idx]
            X_noisy_list.append(noisy_sample); y_noisy_list.append(original_label)
    return np.array(X_noisy_list), np.array(y_noisy_list)

# --- Visualization Functions ---
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
IMAGE_OUTPUT_DIR_NAME = "perceptron_digit_visualizations_7x7"
IMAGE_OUTPUT_DIR = os.path.join(script_dir, IMAGE_OUTPUT_DIR_NAME)
QMD_FILENAME = "perceptron_digit_report_7x7.qmd"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_digit_pattern(pattern_flat, title="Digit Pattern", filename="digit.png"):
    if not MATPLOTLIB_AVAILABLE: return
    pattern_2d = pattern_flat.reshape(GRID_SIZE, GRID_SIZE)
    plt.figure(figsize=(3.5, 3.5)) 
    plt.imshow(pattern_2d, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    plt.title(title, fontsize=10)
    plt.xticks([]); plt.yticks([])
    plt.savefig(os.path.join(IMAGE_OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()

def plot_perceptron_weights(weights_flat, bias, digit_class, filename="weights.png"):
    if not MATPLOTLIB_AVAILABLE: return
    weights_2d = weights_flat.reshape(GRID_SIZE, GRID_SIZE)
    plt.figure(figsize=(4.5, 4.5)) 
    lim = max(abs(weights_2d.min()), abs(weights_2d.max()), 1e-5) 
    plt.imshow(weights_2d, cmap='RdBu', vmin=-lim, vmax=lim, interpolation='nearest')
    plt.colorbar(label="Weight Value")
    # ***** FIX 3 *****
    plt.title(f"Perceptron Weights for Digit '{digit_class}'\nBias: {float(bias):.2f}", fontsize=10)
    plt.xticks([]); plt.yticks([])
    plt.savefig(os.path.join(IMAGE_OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()

def plot_ovr_decision_process(input_pattern_flat, true_label, predicted_label, scores, filename_prefix="decision"):
    if not MATPLOTLIB_AVAILABLE: return
    num_classes = len(scores)
    input_pattern_2d = input_pattern_flat.reshape(GRID_SIZE, GRID_SIZE)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={'width_ratios': [1, 2]})
    axs[0].imshow(input_pattern_2d, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    axs[0].set_title(f"Input Digit (True: {true_label})", fontsize=10)
    axs[0].set_xticks([]); axs[0].set_yticks([])
    # Ensure scores are Python floats for plotting if they came from NumPy
    plot_scores = [float(s) for s in scores]
    axs[1].barh(np.arange(num_classes), plot_scores, color='skyblue')
    axs[1].set_yticks(np.arange(num_classes))
    axs[1].set_yticklabels([f"P ({i})" for i in range(num_classes)])
    axs[1].invert_yaxis() 
    axs[1].set_xlabel("Perceptron Score (Net Input)")
    axs[1].set_title(f"OvR Scores (Predicted: {predicted_label})", fontsize=10)
    axs[1].get_yticklabels()[predicted_label].set_color('red')
    axs[1].get_yticklabels()[predicted_label].set_fontweight('bold')
    if true_label == predicted_label:
        axs[1].barh(predicted_label, plot_scores[predicted_label], color='lightgreen', edgecolor='green')
    else:
        axs[1].barh(predicted_label, plot_scores[predicted_label], color='salmon', edgecolor='red')
    plt.tight_layout()
    filepath = os.path.join(IMAGE_OUTPUT_DIR, f"{filename_prefix}_true{true_label}_pred{predicted_label}.png")
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    return os.path.basename(filepath) 

def plot_network_schematic(filename="network_schematic.png"): # Unchanged
    if not MATPLOTLIB_AVAILABLE: return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')
    input_box = patches.Rectangle((1, 3.5), 2, 1, linewidth=1, edgecolor='black', facecolor='lightgray')
    ax.add_patch(input_box)
    ax.text(2, 4, f"{N_FEATURES} Input Pixels\n({GRID_SIZE}x{GRID_SIZE} Digit)", ha='center', va='center', fontsize=10)
    perceptron_nodes_y = np.linspace(7, 1, 10)
    for i in range(10):
        p_box = patches.Circle((6, perceptron_nodes_y[i]), 0.3, linewidth=1, edgecolor='blue', facecolor='lightblue')
        ax.add_patch(p_box)
        ax.text(6.5, perceptron_nodes_y[i], f"P {i}", ha='left', va='center', fontsize=9)
        ax.plot([3, 5.7], [4, perceptron_nodes_y[i]], 'gray', linestyle='-', linewidth=0.5)
    ax.text(6, 7.5, "Perceptron Units (One-vs-Rest)", ha='center', va='center', fontsize=10)
    output_box = patches.Rectangle((8, 3.5), 1.5, 1, linewidth=1, edgecolor='green', facecolor='lightgreen')
    ax.add_patch(output_box)
    ax.text(8.75, 4, "Max Score\n(Prediction)", ha='center', va='center', fontsize=10)
    if any(isinstance(p, patches.Circle) for p in ax.patches): 
         ax.plot([max([p.center[0]+p.radius for p in ax.patches if isinstance(p, patches.Circle)]) , 8], [np.mean(perceptron_nodes_y),4] , 'gray', linestyle='-', linewidth=0.5)
    plt.title("Conceptual One-vs-Rest Perceptron Network", fontsize=12)
    plt.savefig(os.path.join(IMAGE_OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()
    return filename

# --- QMD Report Generation ---
def generate_qmd_report(
    perfect_test_results, noisy_test_results,
    network_schematic_fn, weights_fns,
    accuracy_perfect, accuracy_noisy,
    num_perfect_samples_shown, num_noisy_samples_shown
):
    qmd_content = f"""---
title: "Perceptron for {GRID_SIZE}x{GRID_SIZE} Digit Recognition"
author: "Automated Report via Python Script"
date: "{datetime.date.today().isoformat()}"
format:
  html:
    toc: true
    code-fold: true
    self-contained: true
jupyter: python3
---

## Introduction
This report demonstrates the use of a Perceptron-based system for recognizing {GRID_SIZE}x{GRID_SIZE} binary pixel digits ({N_FEATURES} features). A One-vs-Rest (OvR) strategy is employed, training ten Perceptrons. Each Perceptron learns to distinguish one specific digit.
The Perceptron is a linear classifier. Key components:
- Input Features: Flattened {GRID_SIZE}x{GRID_SIZE} pixel values.
- Weights and Bias: Learned parameters.
- Net Input: Weighted sum of inputs + bias.
- Activation Function: Step function (0 or 1).
- Learning Rule: Iterative weight/bias adjustment.

## Dataset: {GRID_SIZE}x{GRID_SIZE} Digit Patterns
Base dataset: "perfect" {GRID_SIZE}x{GRID_SIZE} patterns for digits 0-9. Training uses these augmented with noisy versions (randomly flipped pixels).
Example of a perfect digit (Digit 0 if available, else first):
![Perfect Digit Example]({IMAGE_OUTPUT_DIR_NAME}/{os.path.basename(perfect_test_results[0]['input_image_fn']) if perfect_test_results else 'placeholder.png'})

## One-vs-Rest (OvR) Perceptron Model
OvR trains N binary classifiers for N classes (N=10). Perceptron `k` outputs '1' for digit `k`, '0' otherwise. Prediction: highest raw score (net input) from the 10 Perceptrons.

### Conceptual Network Schematic
![Conceptual Network Diagram]({IMAGE_OUTPUT_DIR_NAME}/{network_schematic_fn})

## Training the Perceptrons

The perceptron algorithm starts by initializing all weights to zero or small random values, then repeatedly goes through each training example, calculates the predicted output by taking the weighted sum of inputs plus a bias, and compares this prediction to the actual label. If the prediction matches the label, the weights are left unchanged, but if the prediction is incorrect, the algorithm updates the weights and bias by adding the input vector (scaled by the learning rate and true label) to the weights and adjusting the bias similarly. This process continues, cycling through the data, until all points are correctly classified or a maximum number of passes through the data is reached.

Each Perceptron is trained independently. Visualized weights show important pixels for its target digit (Red: positive, Blue: negative).

More formally: 

1. Initialization:
n_samples, n_features_data = X.shape: Determines the number of training examples and the number of features per example.
self.weights_ = np.zeros(n_features_data): The weights are initialized. Each feature in your input data (e.g., each of the 49 pixels for a 7x7 digit) will have a corresponding weight. Starting with zeros is a common and simple approach, meaning initially, the Perceptron has no preference for any feature. (Alternatively, small random values could be used).
self.bias_ = 0.0: The bias term is initialized to zero. The bias acts like an intercept in a linear equation; it shifts the decision boundary without depending on the input values.
self.errors_ = []: An empty list is created to store the number of misclassifications (errors) made in each epoch. This is useful for monitoring the learning process.
2. Iterative Learning over Epochs:
for epoch in range(self.n_iters):: The algorithm iterates through the entire training dataset multiple times. Each complete pass through the training dataset is called an epoch. self.n_iters defines the maximum number of epochs.
errors_in_epoch = 0: At the start of each epoch, a counter for misclassifications in that epoch is reset.
3. Iterating Through Each Training Sample (Online Learning):
for i in range(n_samples):: Within each epoch, the algorithm processes each training sample one by one.
xi = X[i]: The current training sample's features (e.g., the 49 pixel values for one digit).
target = y[i]: The true class label (0 or 1) for the current training sample xi.
4. Making a Prediction for the Current Sample:
a. Calculate Net Input (Weighted Sum + Bias):
net_input_for_xi = self._net_input(xi)
This internally calls np.dot(xi, self.weights_) + self.bias_.
This is the linear combination of the input features and their corresponding weights, plus the bias. It's a raw score.
z = (w1*x1 + w2*x2 + ... + wn*xn) + b
b. Apply Activation Function:
prediction_activated = self._activation_function(net_input_for_xi)
This internally calls np.where(net_input_for_xi >= 0, 1, 0).
This is a step function. If the net_input_for_xi is greater than or equal to 0, the Perceptron predicts class 1; otherwise, it predicts class 0. This is the Perceptron's actual outputted class label for xi.
5. Calculating the Update Value (Perceptron Learning Rule Core):
update_val = target - prediction_activated: This calculates the error for the current sample.
* If prediction_activated == target (correct classification): update_val will be 0.
* If target = 1 and prediction_activated = 0 (false negative): update_val will be 1 - 0 = 1.
* If target = 0 and prediction_activated = 1 (false positive): update_val will be 0 - 1 = -1.
update = self.learning_rate * update_val:
* self.learning_rate (eta, η): This is a small positive constant (e.g., 0.1, 0.01) that controls the magnitude of the weight and bias adjustments. A smaller learning rate leads to smaller adjustments and potentially slower but more stable convergence. A larger learning rate can lead to faster learning but might overshoot the optimal solution or become unstable.
* The update variable now holds the scaled error. It will be 0 for correct classifications, positive if the prediction should have been 1 but was 0, and negative if the prediction should have been 0 but was 1.
6. Updating Weights and Bias (if Misclassification Occurred):
if update != 0:: The weights and bias are only adjusted if the Perceptron made a mistake on the current sample (update_val was not 0).
a. Update Weights:
self.weights_ += update * xi
This is the core weight update rule. Let's break it down:
* If update is positive (false negative, wanted 1, got 0): self.weights_ will have learning_rate * 1 * xi added to it. For each feature j in xi that was active (e.g., xi[j] == 1), its corresponding weight self.weights_[j] will be increased. This makes it more likely that the net_input will be positive (and thus predict 1) for this sample (or similar samples) in the future.
* If update is negative (false positive, wanted 0, got 1): self.weights_ will have learning_rate * (-1) * xi added to it (i.e., subtracted). For each feature j in xi that was active, self.weights_[j] will be decreased. This makes it more likely that the net_input will be negative (and thus predict 0) for this sample in the future.
* If a feature xi[j] is 0, its corresponding weight self.weights_[j] is not changed by this part of the update, as update * 0 = 0.
b. Update Bias:
self.bias_ += update
* If update is positive (false negative): The bias is increased. Increasing the bias makes it easier for the net_input (weighted sum + bias) to cross the 0 threshold, thus making a prediction of 1 more likely.
* If update is negative (false positive): The bias is decreased, making a prediction of 1 less likely.
errors_in_epoch += 1: Increment the count of misclassifications for the current epoch.
7. Storing Epoch Errors and Checking for Convergence:
After iterating through all samples in an epoch:
self.errors_.append(errors_in_epoch): The total number of misclassifications in that epoch is stored.
if errors_in_epoch == 0 and epoch > 0::
This is an early stopping condition. If an entire epoch completes with zero misclassifications (and it's not the very first epoch which might start with zero errors if weights are initialized to zero and the first sample target is 0), it means the Perceptron has found a set of weights and bias that perfectly separates the training data. The algorithm can then stop, as further iterations won't change the weights.
8. Loop Continuation/Termination:
If convergence is not met and the number of epochs is less than self.n_iters, the algorithm goes back to step 2 for the next epoch.
If self.n_iters is reached, the algorithm stops, even if the data isn't perfectly separated (this happens if the data is not linearly separable or if n_iters is too small).



### Learned Perceptron Weights
"""
    for i in range(10):
        qmd_content += f"""
#### Perceptron for Digit {i}
![Weights for Digit {i}]({IMAGE_OUTPUT_DIR_NAME}/{weights_fns[i]})
*Weights for Perceptron {i} ({N_FEATURES} values reshaped to {GRID_SIZE}x{GRID_SIZE}). Bias noted in title.*
"""

    qmd_content += f"""
## Testing and Results
Model tested on "perfect" patterns and new "noisy" patterns.

### Accuracy
- **Accuracy on Perfect Patterns:** {float(accuracy_perfect):.2f}%
- **Accuracy on New Noisy Patterns:** {float(accuracy_noisy):.2f}%

### Examples of Predictions
Input digit image, then bar chart of scores from 10 Perceptrons. Highest score = prediction.

#### Predictions on Perfect Patterns (showing first {num_perfect_samples_shown})
"""
    for i in range(min(num_perfect_samples_shown, len(perfect_test_results))):
        res = perfect_test_results[i]
        qmd_content += f"""
##### Test Case: Perfect Digit (True: {res['true_label']}, Predicted: {res['predicted_label']})
Input Digit:
![Input Perfect {res['true_label']}]({IMAGE_OUTPUT_DIR_NAME}/{os.path.basename(res['input_image_fn'])})
Decision Process:
![Decision Perfect {res['true_label']}]({IMAGE_OUTPUT_DIR_NAME}/{os.path.basename(res['decision_image_fn'])})
*Scores: {', '.join([f"{float(s):.2f}" for s in res['scores']])}*
---
"""

    qmd_content += f"""
#### Predictions on Noisy Patterns (showing first {num_noisy_samples_shown})
"""
    for i in range(min(num_noisy_samples_shown, len(noisy_test_results))):
        res = noisy_test_results[i]
        qmd_content += f"""
##### Test Case: Noisy Digit (True: {res['true_label']}, Predicted: {res['predicted_label']})
Input Digit:
![Input Noisy {res['true_label']}]({IMAGE_OUTPUT_DIR_NAME}/{os.path.basename(res['input_image_fn'])})
Decision Process:
![Decision Noisy {res['true_label']}]({IMAGE_OUTPUT_DIR_NAME}/{os.path.basename(res['decision_image_fn'])})
*Scores: {', '.join([f"{float(s):.2f}" for s in res['scores']])}*
---
"""

    qmd_content += """
## Conclusion
OvR Perceptrons can classify {GRID_SIZE}x{GRID_SIZE} digits. Performance depends on pattern distinctiveness, noise, learning rate, and iterations. This demonstrates foundational concepts. More complex problems need advanced models. Visualizations offer insight into learning and decision-making.

It's Not Just Looking For the Digit, but Also Against Others (One-vs-Rest):
Remember, the Perceptron for Digit 0 is trained to distinguish "0" (target = 1) from all other digits (target = 0 for digits 1 through 9).
So, the weights are not just learning "what makes a 0 a 0," but also "what makes a 0 not a 1, not a 2, not a 3, etc."
Positive Weights (Red in your RdBu colormap): These pixels, if "on" (value 1) in an input image, push the Perceptron's net input towards classifying the image as a "0". They are features that are strongly indicative of a "0" and/or features that are typically absent in other digits when a "0" is present.
Negative Weights (Blue): These pixels, if "on," push the net input away from classifying it as a "0" (i.e., towards classifying it as "not a 0"). These are features that might be common in other digits but not in a "0", or features whose presence actively contradicts the pattern of a "0".
Near-Zero Weights (White/Light Colors): These pixels don't contribute much to the decision for this specific Perceptron. They might be pixels that are commonly on or off across many digits, or just aren't very discriminative for telling a "0" apart from the rest.

Discriminative Features, Not Perfect Templates:
The Perceptron isn't trying to learn a pixel-perfect template of a "0" that it then matches. Instead, it's learning the most discriminative features.
For example, if the central hole of a "0" is a very strong feature that most other digits don't have, pixels forming that hole might get strong positive weights.
Conversely, if a vertical bar on the far right is very common in a "1" or "7" but never in your "0", those pixels might get strong negative weights for the "0" Perceptron. The presence of these would strongly suggest it's not a "0".

Influence of Other Digits' Shapes:
Let's look at your specific "Weights for Digit 0" image:
You see some strong reddish (positive) weights that seem to roughly outline parts of a "0". This makes sense.
You also see some strong bluish (negative) weights. These might correspond to areas that are typically "on" for other digits but should be "off" for a "0". For instance, if your "1" pattern has a strong central vertical line, the "0" Perceptron might learn negative weights in that central column to "penalize" inputs that look like a "1".
The dark maroon/red spot in the middle of your example image might indicate a pixel that is very reliably "off" in the training examples of 0s, and perhaps "on" in many non-0 digits, thus its absence (being "off" in the input and multiplied by a positive weight, or "on" in the input and multiplied by a negative weight if the colors were reversed for what "on" is) contributes to classifying as a "0".
In your specific image: The dark red patch in the middle left-ish area seems to get a strong positive weight. This means if that pixel is ON, it strongly suggests the digit is a 0. The dark blue patches on the right and left edges (middle rows) have strong negative weights. If these pixels are ON, it strongly suggests the digit is NOT a 0. This could be because digits like '8' or '4' might have pixels active there.
The Bias Term:
The bias term (e.g., "Bias: -0.04" in your image) acts as a general threshold. If the bias is negative, the weighted sum of inputs needs to be even more positive to cross the decision boundary (0) and be classified as a "0". This means the Perceptron is, by default, slightly biased against classifying something as a "0" unless the pixel evidence is strong enough.
Linear Separability and "Good Enough":
A Perceptron finds a linear separating hyperplane. It doesn't necessarily find the "prettiest" or most human-interpretable one. As long as the weighted sum w · x + b correctly pushes "0"s above the threshold and "not 0"s below it for most training examples, the learning algorithm is satisfied for that Perceptron.
The resulting weights are a consequence of the iterative learning process trying to correct misclassifications based on the specific training data (including the noisy examples).


"""
    qmd_file_path = os.path.join(script_dir, QMD_FILENAME)
    with open(qmd_file_path, 'w', encoding='utf-8') as f:
        f.write(qmd_content)
    print(f"\nQuarto Markdown report generated: {qmd_file_path}")
    print(f"To render: `quarto render {QMD_FILENAME}` in '{script_dir}'")



# --- New Combined Visualization Function ---
def plot_combined_overview(
    example_digits_to_show, # List of digit labels (e.g., [0, 1, 7])
    perfect_digit_patterns_flat, # All perfect patterns flattened (X_perfect)
    all_perceptrons_ovr, # List of all 10 trained perceptron models
    filename="combined_overview.png"
):
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for combined overview.")
        return

    num_examples = len(example_digits_to_show)
    if num_examples == 0:
        print("No example digits provided for combined overview.")
        return

    fig = plt.figure(figsize=(6 + num_examples * 3, 8)) # Adjust figsize as needed
    
    # Define GridSpec: 2 main rows. Top row for network, bottom for examples.
    # Bottom row will have 'num_examples' columns for digit+weights pairs.
    gs_main = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5], hspace=0.3, figure=fig)

    # --- Top Part: Network Schematic ---
    ax_network = fig.add_subplot(gs_main[0, 0])
    ax_network.set_xlim(0, 10)
    ax_network.set_ylim(0, 8)
    ax_network.axis('off')
    ax_network.set_title("Conceptual One-vs-Rest (OvR) Network", fontsize=12, pad=10)

    # Input Layer
    input_box_center_x, input_box_center_y = 2, 4
    input_box = patches.Rectangle((input_box_center_x - 1, input_box_center_y - 0.5), 2, 1, lw=1, ec='black', fc='lightgray')
    ax_network.add_patch(input_box)
    ax_network.text(input_box_center_x, input_box_center_y, f"{N_FEATURES} Input\nPixels", ha='center', va='center', fontsize=9)

    # Perceptron Layer
    p_layer_center_x = 6
    perceptron_nodes_y = np.linspace(7, 1, 10)
    p_circles = []
    for i in range(10):
        p_box = patches.Circle((p_layer_center_x, perceptron_nodes_y[i]), 0.3, lw=1, ec='blue', fc='lightblue')
        ax_network.add_patch(p_box)
        p_circles.append(p_box)
        ax_network.text(p_layer_center_x + 0.5, perceptron_nodes_y[i], f" P {i} ", ha='left', va='center', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.1", fc="lightblue", ec="blue", alpha=0.7))
        ax_network.plot([input_box_center_x + 1, p_layer_center_x - 0.3], 
                        [input_box_center_y, perceptron_nodes_y[i]], 
                        'gray', linestyle='-', linewidth=0.5, alpha=0.7)
    
    # Output/Decision
    output_box_center_x, output_box_center_y = 8.75, 4
    output_box = patches.Rectangle((output_box_center_x - 0.75, output_box_center_y - 0.5), 1.5, 1, lw=1, ec='green', fc='lightgreen')
    ax_network.add_patch(output_box)
    ax_network.text(output_box_center_x, output_box_center_y, "Max Score\n(Prediction)", ha='center', va='center', fontsize=9)
    ax_network.plot([p_layer_center_x + 0.3, output_box_center_x - 0.75], 
                    [np.mean(perceptron_nodes_y), output_box_center_y], 
                    'gray', linestyle='-', linewidth=0.5, alpha=0.7)

    # --- Bottom Part: Example Digits and Weights ---
    gs_examples = gridspec.GridSpecFromSubplotSpec(2, num_examples, subplot_spec=gs_main[1, 0], 
                                                   hspace=0.4, wspace=0.3, height_ratios=[1,1.2])

    for i, digit_label in enumerate(example_digits_to_show):
        if digit_label < 0 or digit_label >= len(perfect_digit_patterns_flat) or \
           digit_label >= len(all_perceptrons_ovr):
            print(f"Warning: Digit label {digit_label} is out of bounds. Skipping.")
            continue
            
        # Get the perfect pattern for this digit
        input_pattern_flat = perfect_digit_patterns_flat[digit_label]
        input_pattern_2d = input_pattern_flat.reshape(GRID_SIZE, GRID_SIZE)

        # Get the corresponding Perceptron and its weights
        perceptron_model = all_perceptrons_ovr[digit_label]
        weights_flat = perceptron_model.weights_
        weights_2d = weights_flat.reshape(GRID_SIZE, GRID_SIZE)
        bias = float(perceptron_model.bias_) # Ensure float

        # Subplot for Input Digit
        ax_digit = fig.add_subplot(gs_examples[0, i])
        ax_digit.imshow(input_pattern_2d, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
        ax_digit.set_title(f"Input: Digit '{digit_label}'", fontsize=10)
        ax_digit.set_xticks([])
        ax_digit.set_yticks([])
        
        # Arrow from input network part to this digit example (conceptual)
        # Use figure coordinates for robust arrow placement across subplots
        # This is tricky and might need fine-tuning
        if i == 0: # Draw only for the first example to avoid clutter
            con_input_to_example = patches.ConnectionPatch(
                xyA=(input_box_center_x, input_box_center_y - 1), coordsA=ax_network.transData,
                xyB=(GRID_SIZE/2, GRID_SIZE + 1), coordsB=ax_digit.transData, # Point above the digit plot
                arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=15, fc="gray", ec="gray",
                linestyle="dashed", alpha=0.6
            )
            fig.add_artist(con_input_to_example)


        # Subplot for Weights Matrix
        ax_weights = fig.add_subplot(gs_examples[1, i])
        lim = max(abs(weights_2d.min()), abs(weights_2d.max()), 1e-5)
        im_weights = ax_weights.imshow(weights_2d, cmap='RdBu', vmin=-lim, vmax=lim, interpolation='nearest')
        ax_weights.set_title(f"P({digit_label}) Weights\nBias: {bias:.2f}", fontsize=10)
        ax_weights.set_xticks([])
        ax_weights.set_yticks([])
        
        # Minimal colorbar for weights, try to place it neatly
        # For simplicity, we might omit individual colorbars or add one shared one if space is tight.
        # Let's try a small one for each
        cbar = fig.colorbar(im_weights, ax=ax_weights, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Weight", size=8, labelpad=-15) # Adjust labelpad

        # Arrow from corresponding Perceptron node to its weights (conceptual)
        if digit_label < len(p_circles):
            con_p_to_weights = patches.ConnectionPatch(
                xyA=(p_circles[digit_label].center[0], p_circles[digit_label].center[1]), coordsA=ax_network.transData,
                xyB=(GRID_SIZE/2, GRID_SIZE +1 ), coordsB=ax_weights.transData, # Point above the weights plot
                arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=15, fc="gray", ec="gray",
                linestyle="dashed", alpha=0.6
            )
            fig.add_artist(con_p_to_weights)


    plt.suptitle("Perceptron Digit Recognition: Network, Inputs, and Weights", fontsize=16, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect to make space for suptitle
    
    filepath = os.path.join(IMAGE_OUTPUT_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Combined overview saved to: {filepath}")
    return os.path.basename(filepath)

def animate_perceptron_fitting(perceptron_model, digit_class_label, 
                               grid_size, filename="perceptron_fitting_anim.gif"):
    if not MATPLOTLIB_AVAILABLE or not hasattr(animation, 'FuncAnimation'):
        print("Matplotlib animation tools not available.")
        return

    if not perceptron_model.weight_history_:
        print(f"No weight history found for Perceptron of digit {digit_class_label}. Cannot animate.")
        return

    fig, ax = plt.subplots(figsize=(5, 5)) # Adjust size as needed
    
    # Initial plot setup (first frame)
    initial_weights_flat, initial_bias = perceptron_model.weight_history_[0]
    initial_weights_2d = initial_weights_flat.reshape(grid_size, grid_size)
    
    # Determine a consistent color limit based on the entire history
    all_weights_values = np.concatenate([wh[0] for wh in perceptron_model.weight_history_])
    global_lim = max(abs(all_weights_values.min()), abs(all_weights_values.max()), 1e-5)

    im = ax.imshow(initial_weights_2d, cmap='RdBu', vmin=-global_lim, vmax=global_lim, interpolation='nearest')
    fig.colorbar(im, ax=ax, label="Weight Value")
    title_text = ax.set_title(f"Fitting Perceptron for Digit '{digit_class_label}'\nFrame: 0, Bias: {initial_bias:.2f}")
    ax.set_xticks([])
    ax.set_yticks([])

    num_frames = len(perceptron_model.weight_history_)

    def update_plot(frame_num):
        weights_flat, bias = perceptron_model.weight_history_[frame_num]
        weights_2d = weights_flat.reshape(grid_size, grid_size)
        im.set_data(weights_2d)
        title_text.set_text(f"Fitting Perceptron for Digit '{digit_class_label}'\nFrame: {frame_num+1}/{num_frames}, Bias: {bias:.2f}")
        return [im, title_text]

    # Create animation
    # interval: delay between frames in ms. Adjust for speed.
    # blit=True optimizes drawing but can sometimes cause issues with titles/text.
    anim = animation.FuncAnimation(fig, update_plot, frames=num_frames, 
                                   interval=200, blit=True, repeat=False)

    # Save the animation
    # Needs a writer like 'imagemagick' (for GIF) or 'ffmpeg' (for MP4)
    # Pillow writer can also save GIFs but might be slower or have fewer options.
    writer_engine = None
    if shutil.which('ffmpeg'): # Check if ffmpeg is available
        writer_engine = 'ffmpeg'
        if not filename.endswith(".mp4"): filename = os.path.splitext(filename)[0] + ".mp4"
        print(f"Saving animation as MP4 using ffmpeg to: {os.path.join(IMAGE_OUTPUT_DIR, filename)}")
    elif shutil.which('imagemagick'):
        writer_engine = 'imagemagick'
        if not filename.endswith(".gif"): filename = os.path.splitext(filename)[0] + ".gif"
        print(f"Saving animation as GIF using ImageMagick to: {os.path.join(IMAGE_OUTPUT_DIR, filename)}")
    else: # Fallback to Pillow for GIF
        writer_engine = 'pillow'
        if not filename.endswith(".gif"): filename = os.path.splitext(filename)[0] + ".gif"
        print(f"Saving animation as GIF using Pillow to: {os.path.join(IMAGE_OUTPUT_DIR, filename)}")

    try:
        anim.save(os.path.join(IMAGE_OUTPUT_DIR, filename), writer=writer_engine, dpi=100) # Adjust dpi
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Make sure you have a suitable writer installed (e.g., ImageMagick for GIF, ffmpeg for MP4, or Pillow).")
        print("For Pillow GIFs, you might just need `pip install Pillow`.")
        print("For ImageMagick/ffmpeg, they need to be installed system-wide and in your PATH.")

    plt.close(fig) # Close the figure after saving animation




# --- Main Script ---
if __name__ == "__main__":
    if MATPLOTLIB_AVAILABLE:
        ensure_dir(IMAGE_OUTPUT_DIR)
        print(f"Visualizations will be saved to '{IMAGE_OUTPUT_DIR}/'")
    else:
        print("Matplotlib not available, skipping visualizations and QMD report generation.")
        exit() 
    
    print(f"--- Perceptron for {GRID_SIZE}x{GRID_SIZE} Digit Recognition (One-vs-Rest) ---")

    X_perfect, y_perfect = flatten_patterns(DIGIT_PATTERNS_7x7)
    num_noisy_train_versions = 20 
    train_noise_level = 0.08      
    learning_rate = 0.0002          
    n_iters = 150                 

    X_train, y_train = add_noise(X_perfect, y_perfect, 
                                 num_noisy_versions=num_noisy_train_versions, 
                                 noise_level=train_noise_level)
    permutation = np.random.permutation(len(X_train))
    X_train, y_train = X_train[permutation], y_train[permutation]

    num_classes = len(DIGIT_PATTERNS_7x7)
    perceptrons_ovr = []
    weights_filenames = [None] * num_classes

    print(f"\n--- Training {num_classes} Perceptrons (One-vs-Rest) ---")
    
    
    
    
    training_start_time = time.time()

    # --- Decide which Perceptron to animate (e.g., for digit 0) ---
    DIGIT_TO_ANIMATE = 0 # Choose a digit
    DIGITS_TO_ANIMATE = [0, 1, 2, 3, 4, 5] # Choose a digit
    HISTORY_INTERVAL = 1 # Store weights every 5 updates (misclassifications)

    for i in range(num_classes):
        print(f"Training P{i}...")
        y_bin = np.where(y_train == i, 1, 0)
        p = Perceptron(learning_rate, n_iters, random_state=i)
        
        store_interval_for_this_p = None
        if i in DIGITS_TO_ANIMATE:
            store_interval_for_this_p = HISTORY_INTERVAL
            print(f"  (Will store history for animating P{DIGIT_TO_ANIMATE} every {HISTORY_INTERVAL} updates)")

        p.fit(X_train, y_bin, verbose=False, store_history_interval=store_interval_for_this_p)
        perceptrons_ovr.append(p)
        
        bias = p.bias_; err = p.errors_[-1] if p.errors_ else 'N/A'
        # print(f"  P{i} trained. Bias:{float(bias):.2f}, ErrLastEp:{err}") # Can be verbose
        
        fn=f"weights_digit_{i}.png"; 
        if MATPLOTLIB_AVAILABLE: plot_perceptron_weights(p.weights_,float(bias),i,fn); 
        weights_filenames[i]=fn

    # duration = time.time()-start_time; print(f"--- All trained in {float(duration):.2f}s ---")

        # --- Generate Animation for the chosen Perceptron ---
        if i < len(perceptrons_ovr) and MATPLOTLIB_AVAILABLE:
            perceptron_to_animate = perceptrons_ovr[i]
            if perceptron_to_animate.weight_history_:
                print(f"\n--- Creating animation for Perceptron of Digit '{i}' ---")
                animate_perceptron_fitting(perceptron_to_animate, i, GRID_SIZE, 
                                        filename=f"fitting_anim_digit_{i}.gif") # Or .mp4
        else:
            print(f"No weight history was stored for P{i}, cannot animate.")

    training_duration = time.time() - training_start_time
    # ***** FIX for training_duration formatting *****
    print(f"--- All Perceptrons trained in {float(training_duration):.2f} seconds ---")

    network_schematic_filename = plot_network_schematic(filename="overall_network_schematic_7x7.png")

    perfect_test_results_for_qmd = []
    noisy_test_results_for_qmd = []

    print("\n--- Testing on Perfect Digit Patterns & Visualizing ---")
    correct_predictions_perfect = 0
    num_perfect_to_show_in_qmd = 3 

    for i_sample in range(len(X_perfect)): # Renamed loop variable for clarity
        true_label = y_perfect[i_sample]
        sample_to_predict = X_perfect[i_sample]
        
        # ***** FIX 1 *****
        # _net_input returns a 1-element array; access its first (and only) element.
        raw_scores = [p_model._net_input(np.atleast_2d(sample_to_predict))[0] for p_model in perceptrons_ovr]
        # Convert to list of Python floats for consistent handling
        scores = [float(s) for s in raw_scores] 
        
        predicted_label = np.argmax(scores)
        
        result_str = '(Correct)' if predicted_label == true_label else '(INCORRECT)'
        # ***** FIX 5a *****
        print(f"Digit: {true_label}, Scores: [{', '.join(f'{s:.2f}' for s in scores)}], Predicted: {predicted_label} {result_str}")
        
        if predicted_label == true_label:
            correct_predictions_perfect += 1

        if i_sample < num_perfect_to_show_in_qmd or true_label==0: 
            input_fn = f"perfect_input_true{true_label}_pred{predicted_label}_idx{i_sample}.png"
            plot_digit_pattern(sample_to_predict, 
                               title=f"Input: {true_label}, Pred: {predicted_label}",
                               filename=input_fn)
            
            decision_fn_base = f"decision_perfect_idx{i_sample}"
            decision_fn = plot_ovr_decision_process(sample_to_predict, true_label, predicted_label, scores, 
                                      filename_prefix=decision_fn_base)
            
            perfect_test_results_for_qmd.append({
                'true_label': true_label, 'predicted_label': predicted_label, 'scores': scores,
                'input_image_fn': input_fn, 'decision_image_fn': decision_fn
            })
            
    accuracy_perfect = (correct_predictions_perfect / len(X_perfect)) * 100 if len(X_perfect) > 0 else 0
    # ***** FIX 5b *****
    print(f"\nAccuracy on perfect patterns: {float(accuracy_perfect):.2f}% ({correct_predictions_perfect}/{len(X_perfect)})")

    print("\n--- Testing on a few new Noisy Samples & Visualizing ---")
    num_noisy_to_generate_and_test = 10 
    num_noisy_to_show_in_qmd = 3      
    test_noise_level = 0.12           

    X_test_noisy_base, y_test_noisy_base = flatten_patterns(DIGIT_PATTERNS_7x7)
    X_test_noisy_all, y_test_noisy_all = add_noise(X_test_noisy_base, y_test_noisy_base, 
                                                num_noisy_versions=1, 
                                                noise_level=test_noise_level)
    X_test_noisy_all = X_test_noisy_all[len(X_test_noisy_base):]
    y_test_noisy_all = y_test_noisy_all[len(y_test_noisy_base):]

    if len(X_test_noisy_all) > 0:
        indices = np.arange(len(X_test_noisy_all))
        np.random.shuffle(indices)
        test_indices = indices[:min(num_noisy_to_generate_and_test, len(X_test_noisy_all))] 
    else:
        test_indices = []
        
    correct_predictions_noisy = 0
    actual_noisy_tested_count = len(test_indices)

    for k_idx, original_idx in enumerate(test_indices):
        true_label = y_test_noisy_all[original_idx]
        sample_to_predict = X_test_noisy_all[original_idx]
        
        # ***** FIX 1 (repeated) *****
        raw_scores_noisy = [p_model._net_input(np.atleast_2d(sample_to_predict))[0] for p_model in perceptrons_ovr]
        scores_noisy = [float(s) for s in raw_scores_noisy]

        predicted_label = np.argmax(scores_noisy)
        
        result_str = '(Correct)' if predicted_label == true_label else '(INCORRECT)'
        # ***** FIX 5c *****
        print(f"Noisy Digit (True: {true_label}), Scores: [{', '.join(f'{s:.2f}' for s in scores_noisy)}], Predicted: {predicted_label} {result_str}")
        
        if predicted_label == true_label:
            correct_predictions_noisy += 1

        if k_idx < num_noisy_to_show_in_qmd:
            input_fn = f"noisy_input_true{true_label}_pred{predicted_label}_idx{k_idx}.png"
            plot_digit_pattern(sample_to_predict, 
                               title=f"Noisy Input: {true_label}, Pred: {predicted_label}",
                               filename=input_fn)
            
            decision_fn_base = f"decision_noisy_idx{k_idx}"
            decision_fn = plot_ovr_decision_process(sample_to_predict, true_label, predicted_label, scores_noisy, 
                                      filename_prefix=decision_fn_base)
            
            noisy_test_results_for_qmd.append({
                'true_label': true_label, 'predicted_label': predicted_label, 'scores': scores_noisy,
                'input_image_fn': input_fn, 'decision_image_fn': decision_fn
            })

    accuracy_noisy = (correct_predictions_noisy / actual_noisy_tested_count) * 100 if actual_noisy_tested_count > 0 else 0
    # ***** FIX 5d *****
    print(f"\nAccuracy on {actual_noisy_tested_count} new noisy patterns: {float(accuracy_noisy):.2f}% ({correct_predictions_noisy}/{actual_noisy_tested_count})")

    # Generate QMD Report (FIX 6 is implicitly handled by previous fixes ensuring scores/accuracy are floats)
    generate_qmd_report(
        perfect_test_results=perfect_test_results_for_qmd,
        noisy_test_results=noisy_test_results_for_qmd,
        network_schematic_fn=network_schematic_filename,
        weights_fns=weights_filenames,
        accuracy_perfect=accuracy_perfect, # Already a float
        accuracy_noisy=accuracy_noisy,   # Already a float
        num_perfect_samples_shown=min(num_perfect_to_show_in_qmd, len(perfect_test_results_for_qmd)),
        num_noisy_samples_shown=min(num_noisy_to_show_in_qmd, len(noisy_test_results_for_qmd))
    )
    
    qmd_file_path = os.path.join(script_dir, QMD_FILENAME)
    # --- Automatically Render the Quarto Document ---
    if qmd_file_path and os.path.exists(qmd_file_path):
        quarto_executable = shutil.which("quarto") # Find quarto in PATH
        if quarto_executable:
            print(f"\nAttempting to render Quarto document: {qmd_file_path}")
            try:
                # The command to render
                # `to html` is default but explicit. Output file will be qmd_file_path.html
                command = [quarto_executable, "render", qmd_file_path]
                
                # For a self-contained HTML, Quarto handles it based on the YAML.
                # If you wanted to override or specify output filename:
                # html_output_filename = os.path.splitext(qmd_file_path)[0] + ".html"
                # command.extend(["--to", "html", "--output", html_output_filename])

                result = subprocess.run(command, capture_output=True, text=True, check=False, cwd=script_dir)
                
                if result.returncode == 0:
                    output_html_name = os.path.splitext(os.path.basename(qmd_file_path))[0] + ".html"
                    print(f"Quarto document rendered successfully to {os.path.join(script_dir, output_html_name)}")
                else:
                    print("Error during Quarto rendering:")
                    print("Stdout:", result.stdout)
                    print("Stderr:", result.stderr)
            except FileNotFoundError:
                print(f"Error: Quarto executable not found at '{quarto_executable}'. Please ensure Quarto is installed and in your PATH.")
            except Exception as e:
                print(f"An error occurred while trying to render the Quarto document: {e}")
        else:
            print("Quarto executable not found in PATH. Please render the QMD file manually:")
            print(f"  quarto render {os.path.basename(qmd_file_path)}")
    elif qmd_file_path:
        print(f"QMD file '{qmd_file_path}' was supposed to be generated but not found. Skipping rendering.")
    else:
        print("QMD file generation failed. Skipping rendering.")

    print(f"\nCheck '{IMAGE_OUTPUT_DIR}' for PNGs and '{script_dir}' for the QMD/HTML report.")

    print(f"\nCheck the '{IMAGE_OUTPUT_DIR}' directory for PNG visualizations.")
    
    