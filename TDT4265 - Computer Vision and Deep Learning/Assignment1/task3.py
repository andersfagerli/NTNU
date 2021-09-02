import numpy as np
import utils
import matplotlib.pyplot as plt

from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)

    predictions = model.forward(X)
    num_predictions = predictions.shape[0]

    correct_predictions = np.sum(np.argmax(predictions,axis=1) == np.argmax(targets,axis=1))

    return correct_predictions / num_predictions


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        outputs = self.model.forward(X_batch)
        self.model.backward(X_batch, outputs, Y_batch)

        self.model.w = self.model.w - self.learning_rate * self.model.grad

        return cross_entropy_loss(Y_batch, outputs)

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.05, .25])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.5, .87])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Store weights for lambda=0 (task 4b)
    weights = np.zeros((28*2, 28*trainer.model.num_outputs))
    for i in range(trainer.model.num_outputs):
        weights[:28,(28*i):(28*(i+1))] = np.reshape(trainer.model.w[:-1,i], (28,28))
    
    # Train a model with L2 regularization (task 4b)
    model1 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)

    # Store weights for lambda=1 (task 4b)
    for i in range(trainer.model.num_outputs):
        weights[28:,(28*i):(28*(i+1))] = np.reshape(trainer.model.w[:-1,i], (28,28))
    
    # Plotting of weights (task 4b)
    fig, (ax1,ax2) = plt.subplots(2)
    ax1.imshow(weights[:28,:], cmap="gray")
    ax2.imshow(weights[28:,:], cmap="gray")
    ax1.axis('off')
    ax2.axis('off')
    plt.show()

    # plt.imsave("task4b_softmax_weight_lambda0.pdf", weights[:28,:], cmap="gray", format='pdf')
    # plt.imsave("task4b_softmax_weight_lambda1.pdf", weights[28:,:], cmap="gray", format='pdf')

    # Plotting of accuracy for difference values of lambdas (task 4c)
    
    l2_lambdas = [1, .1, .01, .001]
    ws = np.zeros(len(l2_lambdas))
    count = 0
    for lambd in l2_lambdas:
        model = SoftmaxModel(l2_reg_lambda=lambd)
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history_reg, val_history_reg = trainer.train(num_epochs)

        utils.plot_loss(val_history_reg["accuracy"], r'$\lambda=$'+str(lambd))

        ws[count] = np.linalg.norm(trainer.model.w)

        count += 1
        
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Task 4d - Plotting of the l2 norm for each weight
    plt.plot(l2_lambdas, ws)
    plt.xlabel(r'$\lambda$')
    plt.ylabel("Length")
    plt.show()

