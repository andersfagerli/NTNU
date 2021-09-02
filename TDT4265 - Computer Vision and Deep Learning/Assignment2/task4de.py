import utils
import matplotlib.pyplot as plt
import numpy as np
import time
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    ### Task d) ###
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [60, 60, 10]
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

    # Two hidden layers with 60 hidden units
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    start = time.time()
    train_history, val_history = trainer.train(num_epochs)
    end = time.time()

    print("Elapsed training time (s): ", end-start)

    # Plot accuracy
    plt.figure(figsize=(20, 12))
    plt.ylim([0.90, 1.01])
    utils.plot_loss(train_history["accuracy"], "Training accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Plot loss
    plt.figure(figsize=(20, 12))
    plt.ylim([0.0, 0.4])
    utils.plot_loss(train_history["loss"], "Training loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation loss")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.show()


    ### Task e) ###
    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]

    # 10 hidden layers with 64 hidden units
    model_e = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_e = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_e, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    start = time.time()
    train_history_e, val_history_e = trainer_e.train(num_epochs)
    end = time.time()

    print("Elapsed training time (s): ", end-start)

    # Plot accuracy
    plt.figure(figsize=(20, 12))
    plt.ylim([0.90, 1.01])
    utils.plot_loss(train_history["accuracy"], "Two hidden layers")
    utils.plot_loss(train_history_e["accuracy"], "Ten hidden layers")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Plot loss
    plt.figure(figsize=(20, 12))
    plt.ylim([0.0, 0.4])
    utils.plot_loss(train_history["loss"], "Two hidden layers", npoints_to_average=10)
    utils.plot_loss(train_history_e["loss"], "Ten hidden layers", npoints_to_average=10)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Cross Entropy Loss")
    plt.legend()
    plt.show()