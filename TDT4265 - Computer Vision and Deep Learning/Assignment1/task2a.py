import numpy as np
import utils
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] in the range (-1, 1)
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    X = X * 1/255 * 2 - 1
    return np.c_[X, np.ones(X.shape[0])]


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, 1]
        outputs: outputs of model of shape: [batch size, 1]
    Returns:
        Cross entropy error (float)
    """
    # TODO implement this function (Task 2a)
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    
    return np.mean(-(targets*np.log(outputs)+(1-targets)*np.log(1-outputs)))
    
    


class BinaryModel:

    def __init__(self):
        # Define number of input nodes
        self.I = 785
        self.w = np.zeros((self.I, 1))
        self.grad = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, 1]
        """
        # TODO implement this function (Task 2a)
        return 1.0 / (1.0 + np.exp(-X @ self.w))

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, 1]
            targets: labels/targets of each image of shape: [batch size, 1]
        """
        # TODO implement this function (Task 2a)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        
        N = targets.shape[0]
        self.grad = -1.0/N * X.T @ (targets - outputs)

        assert self.grad.shape == self.w.shape,\
            f"Grad shape: {self.grad.shape}, w: {self.w.shape}"


    def zero_grad(self) -> None:
        self.grad = None


def gradient_approximation_test(model: BinaryModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    w_orig = np.random.normal(loc=0, scale=1/model.w.shape[0]**2, size=model.w.shape)
    epsilon = 1e-3
    for i in range(w_orig.shape[0]):
        model.w = w_orig.copy()
        orig = w_orig[i].copy()
        model.w[i] = orig + epsilon
        logits = model.forward(X)
        cost1 = cross_entropy_loss(Y, logits)
        model.w[i] = orig - epsilon
        logits = model.forward(X)
        cost2 = cross_entropy_loss(Y, logits)
        gradient_approximation = (cost1 - cost2) / (2 * epsilon)
        model.w[i] = orig
        # Actual gradient
        logits = model.forward(X)
        model.backward(X, logits, Y)
        difference = gradient_approximation - model.grad[i, 0]
        assert abs(difference) <= epsilon**2,\
            f"Calculated gradient is incorrect. " \
            f"Approximation: {gradient_approximation}, actual gradient: {model.grad[i,0]}\n" \
            f"If this test fails there could be errors in your cross entropy loss function, " \
            f"forward function or backward function"


if __name__ == "__main__":
    category1, category2 = 2, 3
    X_train, Y_train, *_ = utils.load_binary_dataset(category1, category2)
    X_train = pre_process_images(X_train)
    assert X_train.max() <= 1.0, f"The images (X_train) should be normalized to the range [-1, 1]"
    assert X_train.min() < 0 and X_train.min() >= -1, f"The images (X_train) should be normalized to the range [-1, 1]"
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"
    # Simple test for forward pass. Note that this does not cover all errors!
    model = BinaryModel()
    logits = model.forward(X_train)
    np.testing.assert_almost_equal(
        logits.mean(), .5,
        err_msg="Since the weights are all 0's, the sigmoid activation should be 0.5")

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for i in range(2):
        gradient_approximation_test(model, X_train, Y_train)
        model.w = np.random.randn(*model.w.shape)
