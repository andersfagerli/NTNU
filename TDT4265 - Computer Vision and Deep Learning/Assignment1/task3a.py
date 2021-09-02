import numpy as np
import utils
from task2a import pre_process_images
np.random.seed(1)


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    # TODO implement this function (Task 3a)
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    N = targets.shape[0]
    
    return (-1.0/N) * np.sum(np.sum(targets*np.log(outputs)))

class SoftmaxModel:

    def __init__(self, l2_reg_lambda: float):
        # Define number of input nodes
        self.I = 785

        # Define number of output nodes
        self.num_outputs = 10
        self.w = np.zeros((self.I, self.num_outputs))
        self.grad = None

        self.l2_reg_lambda = l2_reg_lambda

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 3a)
        return np.exp(X @ self.w) / np.sum(np.exp(X @ self.w), axis=1, keepdims=True)
        
    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 3a)
        # To implement L2 regularization task (4b) you can get the lambda value in self.l2_reg_lambda 
        # which is defined in the constructor.
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"

        N = targets.shape[0]
        self.grad = (-1.0/N) * X.T @ (targets - outputs) + 2*self.l2_reg_lambda*self.w

        assert self.grad.shape == self.w.shape,\
             f"Grad shape: {self.grad.shape}, w: {self.w.shape}"

    def zero_grad(self) -> None:
        self.grad = None


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


def gradient_approximation_test(model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    w_orig = np.random.normal(loc=0, scale=1/model.w.shape[0]**2, size=model.w.shape)

    epsilon = 1e-3
    for i in range(model.w.shape[0]):
        for j in range(model.w.shape[1]):
            model.w = w_orig.copy()
            orig = model.w[i, j].copy()
            model.w[i, j] = orig + epsilon
            logits = model.forward(X)
            cost1 = cross_entropy_loss(Y, logits)
            model.w[i, j] = orig - epsilon
            logits = model.forward(X)
            cost2 = cross_entropy_loss(Y, logits)
            gradient_approximation = (cost1 - cost2) / (2 * epsilon)
            model.w[i, j] = orig
            # Actual gradient
            logits = model.forward(X)
            model.backward(X, logits, Y)
            difference = gradient_approximation - model.grad[i, j]
            assert abs(difference) <= epsilon**2,\
                f"Calculated gradient is incorrect. " \
                f"Approximation: {gradient_approximation}, actual gradient: {model.grad[i, j]}\n" \
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
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    # Simple test for forward pass. Note that this does not cover all errors!
    model = SoftmaxModel(0.0)
    logits = model.forward(X_train)
    np.testing.assert_almost_equal(
        logits.mean(), 1/10,
        err_msg="Since the weights are all 0's, the softmax activation should be 1/10")

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for i in range(2):
        gradient_approximation_test(model, X_train, Y_train)
        model.w = np.random.randn(*model.w.shape)
