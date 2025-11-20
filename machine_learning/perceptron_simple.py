import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=100):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights_ = None
        self.bias_ = None
        self.errors_ = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights to small random numbers
        self.weights_ = np.random.random(n_features)
        self.bias_ = 0.0

        print(f"\n--- Training Start ---")
        print(f"Initial weights: {self.weights_}, Initial bias: {self.bias_}")

        for epoch in range(self.n_iters):
            errors_in_epoch = 0
            for i in range(n_samples):
                xi = X[i]
                target = y[i]

                # Calculate net input and predict class
                prediction = self.predict(xi.reshape(1, -1)) # Reshape for single sample

                # Perceptron learning rule
                update = self.learning_rate * (target - prediction[0]) # prediction is an array

                if update != 0: # Misclassification
                    self.weights_ += update * xi
                    self.bias_ += update
                    
                    print(f"Weights updated: {self.weights_}, Bias updated: {self.bias_:.4f}")

    def predict(self, X):
        prediction = np.where(np.dot(X, self.weights_) + self.bias_ >= 0, 1, 0)
        return prediction
    
    
if __name__ == "__main__":
    # We will learn the OR function. if either or both entries of X are 1, y will be 1 ("yes X passes the OR function")
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([ 0, 1, 1, 1])

    # Create the perceptron object
    perceptron = Perceptron(learning_rate=0.1, n_iters=20)
    
    # Fit the perceptron to the training data
    perceptron.fit(X_or, y_or)

    # Test the trained Perceptron
    predictions = perceptron.predict(X_or)
    
    print('actual values:', y_or)
    print('predictions:', predictions)
        
    
