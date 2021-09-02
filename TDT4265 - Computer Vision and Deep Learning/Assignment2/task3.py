import utils
import matplotlib.pyplot as plt
import numpy as np
import time
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = pre_process_images(X_train, mean, std)
    X_val = pre_process_images(X_val, mean, std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created in assignment text - Comparing with and without shuffling.
    # YOU CAN DELETE EVERYTHING BELOW!
    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = True
    learning_rate = 0.02

    new_model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    new_trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        new_model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    start = time.time()
    train_history_new, val_history_new = new_trainer.train(
        num_epochs)
    end = time.time()

    print("Elapsed training time (s): ", end-start)
    
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 2 Model", npoints_to_average=10)
    utils.plot_loss(
        train_history_new["loss"], "Task 2 Model - Improved weight, sigmoid and momentum", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.ylabel("Training Cross Entropy Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.ylim([0.8, .98])
    utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    utils.plot_loss(
        val_history_new["accuracy"], "Task 2 Model - Improved weight, sigmoid and momentum")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()

