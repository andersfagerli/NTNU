import utils
import matplotlib.pyplot as plt
import numpy as np
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = pre_process_images(X_train, mean, std)
    X_val = pre_process_images(X_val, mean, std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # 64 hidden units
    model64 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer64 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model64, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history64, val_history64 = trainer64.train(num_epochs)

    # 32 hidden units
    neurons_per_layer = [32, 10]
    model32 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer32 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model32, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history32, val_history32 = trainer32.train(num_epochs)

    # 128 hidden units
    neurons_per_layer = [128, 10]
    model128 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer128 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model128, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history128, val_history128 = trainer128.train(num_epochs)

    # Plot accuracy
    plt.figure(figsize=(20, 12))
    plt.ylim([0.90, 0.98])
    utils.plot_loss(val_history128["accuracy"], "128 hidden units")
    utils.plot_loss(val_history64["accuracy"], "64 hidden units")
    utils.plot_loss(val_history32["accuracy"], "32 hidden units")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Plot loss
    plt.figure(figsize=(20, 12))
    plt.ylim([0.05, 0.4])
    utils.plot_loss(val_history128["loss"], "128 hidden units")
    utils.plot_loss(val_history64["loss"], "64 hidden units")
    utils.plot_loss(val_history32["loss"], "32 hidden units")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Cross Entropy Loss")
    plt.legend()
    plt.show()