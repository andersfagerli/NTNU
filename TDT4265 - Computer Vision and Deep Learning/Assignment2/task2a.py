import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray, mu=33.55274553571429, sigma=78.87550070784701):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
        mu: mean value of entire training set
        sigma: standard deviation of entire training set
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    X = (X - mu) * 1/sigma
    return np.c_[X, np.ones(X.shape[0])]

def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)
    N = targets.shape[0]
    
    return (-1.0/N) * np.sum(np.sum(targets*np.log(outputs)))


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer
        self.num_layers = len(neurons_per_layer)
        self.Zs = [None for i in range(self.num_layers)]

        # Initialize hidden layers and intermediate activations
        self.num_hidden_layers = len(neurons_per_layer)-1
        self.As = [None for i in range(self.num_hidden_layers)]

        # Initialize the weights and hidden layer outputs
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if use_improved_weight_init:
                w = np.random.normal(0, 1.0 / np.sqrt(prev), w_shape)
            else:
                w = np.random.uniform(-1, 1, w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def sigmoid(self, exponent: np.ndarray) -> np.ndarray:
        """
        Args:
            exponent: exponent in the exponential for the sigmoid activation function
        Returns:
            a: activation after pass through sigmoid
        """
        return 1.0 / (1 + np.exp(-exponent))
    
    def dsigmoid(self, exponent: np.ndarray) -> np.ndarray:
        """
        Args:
            exponent: exponent in the exponential for the differientiated sigmoid function
        Returns:
            da: activation derivative after pass through differientiated sigmoid function
        """
        return self.sigmoid(-exponent) * (1.0 - self.sigmoid(-exponent))

    def improved_sigmoid(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            x: argument in the hyperbolic tangent for the improved sigmoid function
        Returns:
            a: activation after pass through improved sigmoid
        """
        return 1.7159 * np.tanh(2.0/3 * X)

    def improved_dsigmoid(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            x: argument in the hyperbolic tangent for the differientiated improved sigmoid function
        Returns:
            da: activation derivative after pass through differientiated improved sigmoid
        """
        return (1.7159 * 4.0/3) / (np.cosh((4.0/3) * X) + 1)
    
    def activation(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: argument to pass through chosen activation function
        Returns:
            a: activation after pass through chosen activation function
        """
        if self.use_improved_sigmoid:
            return self.improved_sigmoid(X)
        else:
            return self.sigmoid(X)
        
    def dactivation(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: argument to pass through chosen activation function derivative
        Returns:
            da: activation after pass through chosen activation function derivative
        """
        if self.use_improved_sigmoid:
            return self.improved_dsigmoid(X)
        else:
            return self.dsigmoid(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For peforming the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.As = ...
        self.Zs[0] = X @ self.ws[0]
        A = self.activation(self.Zs[0])
        for layer in range(self.num_hidden_layers):
            self.As[layer] = A
            self.Zs[layer+1] = A @ self.ws[layer+1]
            A = self.activation(self.Zs[layer+1])
        return np.exp(self.Zs[-1]) / np.sum(np.exp(self.Zs[-1]), axis=1, keepdims=True)

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"

        N = targets.shape[0]
        delta_k = -(targets - outputs)

        if self.num_hidden_layers >= 1:
            # Output to hidden backpropagation
            self.grads[-1] = (1.0/N) * self.As[-1].T @ delta_k

            # Hidden to hidden backpropagation
            delta_j = delta_k
            for j in reversed(range(1, self.num_layers-1)):
                delta_j = self.dactivation(self.Zs[j]) * (delta_j @ self.ws[j+1].T)
                self.grads[j] = (1.0/N) * self.As[j-1].T @ delta_j
            
            # Hidden to input backpropagation
            delta_j = self.dactivation(self.Zs[0]) * (delta_j @ self.ws[1].T)
            self.grads[0] = (1.0/N) * X.T @ delta_j
        else:
            # No hidden layers
            self.grads[-1] = (1.0/N) * X.T @ delta_k
        
        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    Y_reshaped = np.copy(Y).reshape(-1)
    
    return np.eye(num_classes)[Y_reshaped]


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    
    mean = np.mean(X_train)
    std = np.std(X_train)
    print("Mean of training set: ", mean)
    print("Std of training set: ", std)
    
    X_train = pre_process_images(X_train, mean, std)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
