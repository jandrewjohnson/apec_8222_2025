import numpy as np
import time
import os # For creating directories

# --- Matplotlib for Visualization ---
# This is an external library, not strictly NumPy/SciPy, but essential for good visualization.
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: Matplotlib not found. Visualizations will not be generated.")
    print("Please install Matplotlib: pip install matplotlib")

# --- Perceptron Class (same as before) ---
class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=100, random_state=None):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        if random_state:
            np.random.seed(random_state)
        self.weights_ = None
        self.bias_ = None
        self.errors_ = []

    def _activation_function(self, x):
        return np.where(x >= 0, 1, 0)

    def _net_input(self, X):
        X_proc = np.atleast_2d(X)
        return np.dot(X_proc, self.weights_) + self.bias_

    def fit(self, X, y, verbose=False):
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.errors_ = []

        if verbose:
            print(f"  Initial bias: {self.bias_}")

        for epoch in range(self.n_iters):
            errors_in_epoch = 0
            for i in range(n_samples):
                xi = X[i]
                target = y[i]
                prediction_activated = self._activation_function(self._net_input(xi))
                update = self.learning_rate * (target - prediction_activated)
                if update != 0:
                    self.weights_ += update * xi
                    self.bias_ += update
                    errors_in_epoch += 1
            
            self.errors_.append(errors_in_epoch)
            if verbose and (epoch < 3 or epoch % (self.n_iters // 10 or 1) == 0 or errors_in_epoch == 0):
                 print(f"    Epoch {epoch+1}/{self.n_iters} - Updates: {errors_in_epoch}, Bias: {self.bias_:.4f}")
            
            if errors_in_epoch == 0 and epoch > 0:
                if verbose:
                    print(f"    Converged at epoch {epoch+1}!")
                break
        if verbose:
            print(f"  Final bias: {self.bias_:.4f}")
        return self

    def predict(self, X):
        return self._activation_function(self._net_input(X)).flatten()

# --- Digit Patterns and Helper Functions (same as before) ---
DIGIT_PATTERNS = {
    0: np.array([[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1]]),
    1: np.array([[0,0,1,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0]]),
    2: np.array([[1,1,1,1,0],[0,0,0,0,1],[0,1,1,1,0],[1,0,0,0,0],[1,1,1,1,1]]),
    3: np.array([[1,1,1,1,0],[0,0,0,0,1],[0,1,1,1,1],[0,0,0,0,1],[1,1,1,1,0]]),
    4: np.array([[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1],[0,0,0,0,1],[0,0,0,0,1]]),
    5: np.array([[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,0],[0,0,0,0,1],[1,1,1,1,0]]),
    6: np.array([[0,1,1,1,0],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,1],[0,1,1,1,0]]),
    7: np.array([[1,1,1,1,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,0,1,0,0]]),
    8: np.array([[0,1,1,1,0],[1,0,0,0,1],[0,1,1,1,0],[1,0,0,0,1],[0,1,1,1,0]]),
    9: np.array([[0,1,1,1,0],[1,0,0,0,1],[0,1,1,1,1],[0,0,0,0,1],[0,1,1,1,0]])
}

def flatten_patterns(patterns_dict):
    X_list, y_list = [], []
    for digit, pattern in patterns_dict.items():
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
            flip_indices = np.random.choice(n_features, size=num_flips, replace=False)
            for idx in flip_indices: noisy_sample[idx] = 1 - noisy_sample[idx]
            X_noisy_list.append(noisy_sample); y_noisy_list.append(original_label)
    return np.array(X_noisy_list), np.array(y_noisy_list)

# --- Visualization Functions ---
script_dir = os.path.dirname(os.path.abspath(__file__))
IMAGE_OUTPUT_DIR = os.path.join(script_dir, "perceptron_digit_visualizations")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_digit_pattern(pattern_flat, title="Digit Pattern", filename="digit.png"):
    if not MATPLOTLIB_AVAILABLE: return
    pattern_2d = pattern_flat.reshape(5, 5)
    plt.figure(figsize=(3, 3))
    plt.imshow(pattern_2d, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(IMAGE_OUTPUT_DIR, filename))
    plt.close()

def plot_perceptron_weights(weights_flat, bias, digit_class, filename="weights.png"):
    if not MATPLOTLIB_AVAILABLE: return
    weights_2d = weights_flat.reshape(5, 5)
    plt.figure(figsize=(4, 4))
    # Use a diverging colormap to show positive and negative weights
    lim = max(abs(weights_2d.min()), abs(weights_2d.max())) # Symmetrical color scale
    plt.imshow(weights_2d, cmap='RdBu', vmin=-lim if lim >0 else -1, vmax=lim if lim >0 else 1, interpolation='nearest')
    plt.colorbar(label="Weight Value")
    plt.title(f"Perceptron Weights for Digit '{digit_class}'\nBias: {bias}")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(IMAGE_OUTPUT_DIR, filename))
    plt.close()

def plot_ovr_decision_process(input_pattern_flat, true_label, predicted_label, scores, filename_prefix="decision"):
    if not MATPLOTLIB_AVAILABLE: return
    
    num_classes = len(scores)
    input_pattern_2d = input_pattern_flat.reshape(5, 5)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 2]})

    # Plot Input Digit
    axs[0].imshow(input_pattern_2d, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    axs[0].set_title(f"Input Digit (True: {true_label})")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # Plot Perceptron Scores and Decision
    axs[1].barh(np.arange(num_classes), scores, color='skyblue')
    axs[1].set_yticks(np.arange(num_classes))
    axs[1].set_yticklabels([f"P ({i})" for i in range(num_classes)])
    axs[1].invert_yaxis() # Display 0 at top
    axs[1].set_xlabel("Perceptron Score (Net Input)")
    axs[1].set_title(f"OvR Scores (Predicted: {predicted_label})")
    
    # Highlight the winning perceptron
    axs[1].get_yticklabels()[predicted_label].set_color('red')
    axs[1].get_yticklabels()[predicted_label].set_fontweight('bold')
    if true_label == predicted_label:
        axs[1].barh(predicted_label, scores[predicted_label], color='lightgreen', edgecolor='green')
    else:
        axs[1].barh(predicted_label, scores[predicted_label], color='salmon', edgecolor='red')


    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_OUTPUT_DIR, f"{filename_prefix}_true{true_label}_pred{predicted_label}.png"))
    plt.close()

def plot_network_schematic(filename="network_schematic.png"):
    if not MATPLOTLIB_AVAILABLE: return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Input Layer (conceptual)
    input_box = patches.Rectangle((1, 3.5), 2, 1, linewidth=1, edgecolor='black', facecolor='lightgray')
    ax.add_patch(input_box)
    ax.text(2, 4, "25 Input Pixels\n(5x5 Digit)", ha='center', va='center', fontsize=10)

    # Perceptron Layer
    perceptron_nodes_y = np.linspace(7, 1, 10)
    for i in range(10):
        p_box = patches.Circle((6, perceptron_nodes_y[i]), 0.3, linewidth=1, edgecolor='blue', facecolor='lightblue')
        ax.add_patch(p_box)
        ax.text(6.5, perceptron_nodes_y[i], f"P {i}", ha='left', va='center', fontsize=9)
        # Connections (conceptual)
        ax.plot([3, 5.7], [4, perceptron_nodes_y[i]], 'gray', linestyle='-', linewidth=0.5)

    ax.text(6, 7.5, "Perceptron Units (One-vs-Rest)", ha='center', va='center', fontsize=10)
    
    # Output/Decision (conceptual)
    output_box = patches.Rectangle((8, 3.5), 1.5, 1, linewidth=1, edgecolor='green', facecolor='lightgreen')
    ax.add_patch(output_box)
    ax.text(8.75, 4, "Max Score\n(Prediction)", ha='center', va='center', fontsize=10)
    ax.plot([max([p.center[0]+p.radius for p in ax.patches if isinstance(p, patches.Circle)]) , 8], [4,4] , 'gray', linestyle='-', linewidth=0.5)


    plt.title("Conceptual One-vs-Rest Perceptron Network", fontsize=12)
    plt.savefig(os.path.join(IMAGE_OUTPUT_DIR, filename))
    plt.close()


# --- Main Script ---
if __name__ == "__main__":
    if MATPLOTLIB_AVAILABLE:
        ensure_dir(IMAGE_OUTPUT_DIR)
        print(f"Visualizations will be saved to '{IMAGE_OUTPUT_DIR}/'")
    
    print("--- Perceptron for 5x5 Digit Recognition (One-vs-Rest) ---")

    X_perfect, y_perfect = flatten_patterns(DIGIT_PATTERNS)
    X_train, y_train = add_noise(X_perfect, y_perfect, num_noisy_versions=15, noise_level=0.10)
    permutation = np.random.permutation(len(X_train))
    X_train, y_train = X_train[permutation], y_train[permutation]

    num_classes = len(DIGIT_PATTERNS)
    perceptrons_ovr = []
    learning_rate = 0.05 # Adjusted
    n_iters = 100      # Adjusted

    print(f"\n--- Training {num_classes} Perceptrons (One-vs-Rest) ---")
    training_start_time = time.time()
    for i in range(num_classes):
        print(f"Training Perceptron for digit '{i}' vs Rest...")
        y_binary_target = np.where(y_train == i, 1, 0)
        perceptron = Perceptron(learning_rate=learning_rate, n_iters=n_iters, random_state=i)
        perceptron.fit(X_train, y_binary_target, verbose=False)
        perceptrons_ovr.append(perceptron)
        print(f"  Perceptron for digit '{i}' trained. Bias: {perceptron.bias_}, Errors last epoch: {perceptron.errors_[-1] if perceptron.errors_ else 'N/A'}")
        
        # Plot weights for this perceptron
        plot_perceptron_weights(perceptron.weights_, perceptron.bias_, i, filename=f"weights_digit_{i}.png")

    training_duration = time.time() - training_start_time
    print(f"--- All Perceptrons trained in {training_duration:.2f} seconds ---")

    # Plot the conceptual network diagram once
    plot_network_schematic(filename="overall_network_schematic.png")

    print("\n--- Testing on Perfect Digit Patterns & Visualizing ---")
    correct_predictions_perfect = 0
    for i in range(len(X_perfect)): # Test on a few perfect patterns
        true_label = y_perfect[i]
        sample_to_predict = X_perfect[i]
        
        scores = [p_model._net_input(np.atleast_2d(sample_to_predict))[0] for p_model in perceptrons_ovr]
        predicted_label = np.argmax(scores)
        
        result_str = '(Correct)' if predicted_label == true_label else '(INCORRECT)'
        print(f"Digit: {true_label}, Scores: [{', '.join(f'{s:.2f}' for s in scores)}], Predicted: {predicted_label} {result_str}")
        
        if predicted_label == true_label:
            correct_predictions_perfect += 1

        # Plot input pattern
        plot_digit_pattern(sample_to_predict, 
                           title=f"Input: {true_label}, Predicted: {predicted_label}",
                           filename=f"perfect_input_true{true_label}_pred{predicted_label}_idx{i}.png")
        # Plot decision process for this sample
        plot_ovr_decision_process(sample_to_predict, true_label, predicted_label, scores, 
                                  filename_prefix=f"decision_perfect_idx{i}")
            
    accuracy_perfect = (correct_predictions_perfect / len(X_perfect)) * 100
    print(f"\nAccuracy on perfect patterns: {accuracy_perfect:.2f}% ({correct_predictions_perfect}/{len(X_perfect)})")

    print("\n--- Testing on a few new Noisy Samples & Visualizing ---")
    num_test_noisy = 5 # Generate a few noisy samples for detailed visualization
    # Generate distinct noisy samples for testing
    X_test_noisy_base, y_test_noisy_base = flatten_patterns(DIGIT_PATTERNS) # Use perfect as base
    X_test_noisy, y_test_noisy = add_noise(X_test_noisy_base, y_test_noisy_base, num_noisy_versions=1, noise_level=0.15)
    
    # Ensure we only pick one noisy version per original digit for unique testing if num_noisy_versions > 0
    # This logic takes the original, then one noisy version for each.
    # Let's pick a random subset of these for testing to keep output manageable.
    
    indices = np.arange(len(X_test_noisy))
    np.random.shuffle(indices)
    test_indices = indices[:num_test_noisy]

    correct_predictions_noisy = 0
    for k_idx, original_idx in enumerate(test_indices): # Use k_idx for unique filenames
        true_label = y_test_noisy[original_idx]
        sample_to_predict = X_test_noisy[original_idx]
        
        scores = [p_model._net_input(np.atleast_2d(sample_to_predict))[0] for p_model in perceptrons_ovr]
        predicted_label = np.argmax(scores)
        
        result_str = '(Correct)' if predicted_label == true_label else '(INCORRECT)'
        print(f"Noisy Digit (True: {true_label}), Scores: [{', '.join(f'{s}' for s in scores)}], Predicted: {predicted_label} {result_str}")
        
        if predicted_label == true_label:
            correct_predictions_noisy += 1

        plot_digit_pattern(sample_to_predict, 
                           title=f"Noisy Input: {true_label}, Predicted: {predicted_label}",
                           filename=f"noisy_input_true{true_label}_pred{predicted_label}_idx{k_idx}.png")
        plot_ovr_decision_process(sample_to_predict, true_label, predicted_label, scores, 
                                  filename_prefix=f"decision_noisy_idx{k_idx}")

    if num_test_noisy > 0:
        accuracy_noisy = (correct_predictions_noisy / num_test_noisy) * 100
        print(f"\nAccuracy on {num_test_noisy} new noisy patterns: {accuracy_noisy:.2f}% ({correct_predictions_noisy}/{num_test_noisy})")
    else:
        print("\nNo new noisy patterns generated for detailed testing visualization.")

    print(f"\nCheck the '{IMAGE_OUTPUT_DIR}' directory for PNG visualizations.")