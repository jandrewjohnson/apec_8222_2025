import numpy as np
import time # To see training time

class Perceptron:
    """
    A simple Perceptron classifier.
    (Slightly modified for OvR usage - predict method will be less relevant,
     we'll use _net_input directly for OvR scoring)
    """
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
        # Ensure X is 2D for np.dot if it's a single sample
        X_proc = np.atleast_2d(X)
        return np.dot(X_proc, self.weights_) + self.bias_

    def fit(self, X, y, verbose=False):
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features) # Initialize weights to zeros
        # self.weights_ = np.random.rand(n_features) * 0.01 # Or small randoms
        self.bias_ = 0.0
        self.errors_ = []

        if verbose:
            print(f"  Initial weights: {self.weights_[:3]}... (len {len(self.weights_)}), Initial bias: {self.bias_}")

        for epoch in range(self.n_iters):
            errors_in_epoch = 0
            for i in range(n_samples):
                xi = X[i]
                target = y[i]
                
                # Predict using current weights
                # For direct use, we want the output of activation
                prediction_activated = self._activation_function(self._net_input(xi))

                update = self.learning_rate * (target - prediction_activated)

                if update != 0: # Misclassification
                    self.weights_ += update * xi
                    self.bias_ += update
                    errors_in_epoch += 1
            
            self.errors_.append(errors_in_epoch)
            if verbose and (epoch < 3 or epoch % (self.n_iters // 10 or 1) == 0 or errors_in_epoch == 0):
                 print(f"    Epoch {epoch+1}/{self.n_iters} - Updates: {errors_in_epoch}, Bias: {self.bias_:.4f}")
            
            if errors_in_epoch == 0 and epoch > 0: # Converged if no errors (and not first epoch)
                if verbose:
                    print(f"    Converged at epoch {epoch+1}!")
                break
        if verbose:
            print(f"  Final bias: {self.bias_:.4f}")
        return self

    def predict(self, X):
        """Return class label after unit step."""
        return self._activation_function(self._net_input(X)).flatten() # ensure 1D output


# --- Define 5x5 Pixel Representations of Digits ---
# 1 for pixel ON, 0 for pixel OFF
DIGIT_PATTERNS = {
    0: np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]),
    1: np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0]
    ]),
    2: np.array([
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1]
    ]),
    3: np.array([
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0]
    ]),
    4: np.array([
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1]
    ]),
    5: np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0]
    ]),
    6: np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0]
    ]),
    7: np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ]),
    8: np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0]
    ]),
    9: np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0]
    ])
}

def flatten_patterns(patterns_dict):
    """Flattens 5x5 patterns into 1D vectors and creates labels."""
    X_list = []
    y_list = []
    for digit, pattern in patterns_dict.items():
        X_list.append(pattern.flatten())
        y_list.append(digit)
    return np.array(X_list), np.array(y_list)

def add_noise(X_original, y_original, num_noisy_versions=5, noise_level=0.05):
    """Creates noisy versions of the original patterns."""
    X_noisy_list = []
    y_noisy_list = []
    
    n_features = X_original.shape[1]
    
    for i in range(X_original.shape[0]):
        original_sample = X_original[i]
        original_label = y_original[i]
        
        # Add the original sample
        X_noisy_list.append(original_sample)
        y_noisy_list.append(original_label)
        
        for _ in range(num_noisy_versions):
            noisy_sample = original_sample.copy()
            # Flip a small percentage of pixels
            num_flips = int(noise_level * n_features)
            flip_indices = np.random.choice(n_features, size=num_flips, replace=False)
            for idx in flip_indices:
                noisy_sample[idx] = 1 - noisy_sample[idx] # Flip 0 to 1, 1 to 0
            X_noisy_list.append(noisy_sample)
            y_noisy_list.append(original_label)
            
    return np.array(X_noisy_list), np.array(y_noisy_list)


# --- Main Script ---
if __name__ == "__main__":
    print("--- Perceptron for 5x5 Digit Recognition (One-vs-Rest) ---")

    # 1. Prepare Data
    X_perfect, y_perfect = flatten_patterns(DIGIT_PATTERNS)
    print(f"\nPerfect patterns created. X_perfect shape: {X_perfect.shape}, y_perfect shape: {y_perfect.shape}")

    # Add noisy versions for more robust training
    # Increase num_noisy_versions for more data, might need more n_iters then
    X_train, y_train = add_noise(X_perfect, y_perfect, num_noisy_versions=10, noise_level=0.08) 
    print(f"Training data with noise. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    # Shuffle the training data (good practice)
    permutation = np.random.permutation(len(X_train))
    X_train = X_train[permutation]
    y_train = y_train[permutation]

    # 2. Train One Perceptron per Digit (One-vs-Rest)
    num_classes = len(DIGIT_PATTERNS)
    perceptrons_ovr = []
    learning_rate = 0.1
    n_iters = 50 # Increase if convergence is slow or data is very noisy/complex
    
    print(f"\n--- Training {num_classes} Perceptrons (One-vs-Rest) ---")
    training_start_time = time.time()

    for i in range(num_classes): # For digit 0, 1, ..., 9
        print(f"Training Perceptron for digit '{i}' vs Rest...")
        
        # Create binary labels for the current Perceptron:
        # 1 if it's the current digit, 0 otherwise
        y_binary_target = np.where(y_train == i, 1, 0)
        
        perceptron = Perceptron(learning_rate=learning_rate, n_iters=n_iters, random_state=i) # Different random state for variety
        # Pass verbose=True to see epoch details for each perceptron, False for less output
        perceptron.fit(X_train, y_binary_target, verbose=False) 
        perceptrons_ovr.append(perceptron)
        # print(f"  Perceptron for digit '{i}' trained. Final bias: {perceptron.bias_:.3f}, Errors in last epoch: {perceptron.errors_[-1] if perceptron.errors_ else 'N/A'}")

    training_duration = time.time() - training_start_time
    print(f"--- All Perceptrons trained in {training_duration:.2f} seconds ---")

    # 3. Test the OvR Classifier
    # We will test on the original "perfect" patterns and some noisy ones
    
    print("\n--- Testing on Perfect Digit Patterns ---")
    correct_predictions_perfect = 0
    for i in range(len(X_perfect)):
        true_label = y_perfect[i]
        sample_to_predict = X_perfect[i]
        
        scores = []
        for p_idx, p_model in enumerate(perceptrons_ovr):
            # Use the raw net input as the score
            # Add np.atleast_2d because _net_input expects 2D X
            score = p_model._net_input(np.atleast_2d(sample_to_predict))[0] # Get scalar score
            scores.append(score)
        
        predicted_label = np.argmax(scores)
        
        print(f"Digit: {true_label}, Scores: [{', '.join(f'{s:.2f}' for s in scores)}], Predicted: {predicted_label} {'(Correct)' if predicted_label == true_label else '(INCORRECT)'}")
        if predicted_label == true_label:
            correct_predictions_perfect += 1
            
    accuracy_perfect = (correct_predictions_perfect / len(X_perfect)) * 100
    print(f"\nAccuracy on perfect patterns: {accuracy_perfect:.2f}% ({correct_predictions_perfect}/{len(X_perfect)})")


    print("\n--- Testing on a few new Noisy Samples (generated on the fly) ---")
    num_test_noisy = 10
    X_test_noisy, y_test_noisy = add_noise(X_perfect, y_perfect, num_noisy_versions=0, noise_level=0.12) # just one version per original, but noisy
    # Shuffle test noisy
    perm_noisy = np.random.permutation(len(X_test_noisy))
    X_test_noisy = X_test_noisy[perm_noisy][:num_test_noisy]
    y_test_noisy = y_test_noisy[perm_noisy][:num_test_noisy]


    correct_predictions_noisy = 0
    for i in range(len(X_test_noisy)):
        true_label = y_test_noisy[i]
        sample_to_predict = X_test_noisy[i]
        
        scores = []
        for p_idx, p_model in enumerate(perceptrons_ovr):
            score = p_model._net_input(np.atleast_2d(sample_to_predict))[0]
            scores.append(score)
        
        predicted_label = np.argmax(scores)
        
        # Optionally visualize the noisy input
        # print(f"\nInput (True: {true_label}):")
        # for row_idx in range(5):
        #     print(" ".join(['#' if x == 1 else '.' for x in sample_to_predict[row_idx*5:(row_idx+1)*5]]))

        print(f"Noisy Digit (True: {true_label}), Scores: [{', '.join(f'{s:.2f}' for s in scores)}], Predicted: {predicted_label} {'(Correct)' if predicted_label == true_label else '(INCORRECT)'}")
        if predicted_label == true_label:
            correct_predictions_noisy += 1

    if len(X_test_noisy) > 0:
        accuracy_noisy = (correct_predictions_noisy / len(X_test_noisy)) * 100
        print(f"\nAccuracy on {len(X_test_noisy)} new noisy patterns: {accuracy_noisy:.2f}% ({correct_predictions_noisy}/{len(X_test_noisy)})")
    else:
        print("\nNo new noisy patterns generated for testing.")

    print("\nNote: The 'scores' are the raw net_input (w.x + b) from each Perceptron.")
    print("The digit corresponding to the Perceptron with the highest score is chosen.")
    print("If patterns are too similar or noise is too high, misclassifications can occur.")
    print("Try adjusting `n_iters`, `learning_rate`, `num_noisy_versions`, and `noise_level`.")