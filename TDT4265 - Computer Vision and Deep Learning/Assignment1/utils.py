from typing import Generator
import mnist
import numpy as np
import matplotlib.pyplot as plt


def batch_loader(
        X: np.ndarray, Y: np.ndarray,
        batch_size: int, shuffle=False,
        drop_last=True) -> Generator:
    """
    Creates a batch generator over the whole dataset (X, Y) which returns a generator iterating over all the batches.
    This function is called once each epoch.
    Often drop_last is set to True for the train dataset, but not for the train set.

    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
        Y: labels of shape [batch size]
        drop_last: Drop last batch if len(X) is not divisible by batch size
        shuffle (bool): To shuffle the dataset between each epoch or not.
    """
    assert len(X) == len(Y)
    num_batches = len(X) // batch_size
    if not drop_last:
        num_batches = int(np.ceil(len(X) / batch_size))
    indices = list(range(len(X)))

    # TODO (task 2e) implement dataset shuffling here.
    if shuffle:
        np.random.shuffle(indices)

    for i in range(num_batches):
        # select a set of indices for each batch of samples
        batch_indices = indices[i*batch_size:(i+1)*batch_size]
        x = X[batch_indices]
        y = Y[batch_indices]
        # return both images (x) and labels (y)
        yield (x, y, batch_indices)


### NO NEED TO EDIT ANY CODE BELOW THIS ###

def binary_prune_dataset(class1: int, class2: int,
                         X: np.ndarray, Y: np.ndarray):
    """
    Splits the dataset into the class 1 and class2. All other classes are removed.
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
        Y: labels of shape [batch size]
    """

    mask1 = (Y == class1)
    mask2 = (Y == class2)
    mask_total = np.bitwise_or(mask1, mask2)
    Y_binary = Y.copy()
    Y_binary[mask1] = 1
    Y_binary[mask2] = 0
    return X[mask_total], Y_binary[mask_total]


def load_binary_dataset(class1: int, class2: int):
    """
    Loads, prunes and splits the dataset into train, and validation.
    """
    train_size = 20000
    val_size = 10000
    X_train, Y_train, X_val, Y_val = mnist.load()

    # First 20000 images from train set
    X_train, Y_train = X_train[:train_size], Y_train[:train_size]
    # Last 2000 images from test set
    X_val, Y_val = X_val[:val_size], Y_val[:val_size]
    X_train, Y_train = binary_prune_dataset(
        class1, class2, X_train, Y_train
    )
    X_val, Y_val = binary_prune_dataset(
        class1, class2, X_val, Y_val
    )
    # Reshape to (N, 1)
    Y_train = Y_train.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)

    print(f"Train shape: X: {X_train.shape}, Y: {Y_train.shape}")
    print(f"Validation shape: X: {X_val.shape}, Y: {Y_val.shape}")

    return X_train, Y_train, X_val, Y_val


def load_full_mnist():
    """
    Loads and splits the dataset into train, validation and test.
    """
    train_size = 20000
    test_size = 10000
    X_train, Y_train, X_val, Y_val = mnist.load()

    # First 20000 images from train set
    X_train, Y_train = X_train[:train_size], Y_train[:train_size]
    # Last 2000 images from test set
    X_val, Y_val = X_val[-test_size:], Y_val[-test_size:]
    # Reshape to (N, 1)
    Y_train = Y_train.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)

    print(f"Train shape: X: {X_train.shape}, Y: {Y_train.shape}")
    print(f"Validation shape: X: {X_val.shape}, Y: {Y_val.shape}")

    return X_train, Y_train, X_val, Y_val


def plot_loss(loss_dict: dict, label: str = None, npoints_to_average=1, plot_variance=True):
    """
    Args:
        loss_dict: a dictionary where keys are the global step and values are the given loss / accuracy
        label: a string to use as label in plot legend
        npoints_to_average: Number of points to average plot
    """
    global_steps = list(loss_dict.keys())
    loss = list(loss_dict.values())
    if npoints_to_average == 1 or not plot_variance:
        plt.plot(global_steps, loss, label=label)
        return

    npoints_to_average = 10
    num_points = len(loss) // npoints_to_average
    mean_loss = []
    loss_std = []
    steps = []
    for i in range(num_points):
        points = loss[i*npoints_to_average:(i+1)*npoints_to_average]
        step = global_steps[i*npoints_to_average + npoints_to_average//2]
        mean_loss.append(np.mean(points))
        loss_std.append(np.std(points))
        steps.append(step)
    plt.plot(steps, mean_loss,
             label=f"{label} mean over {npoints_to_average} steps")
    plt.fill_between(
        steps, np.array(mean_loss) -
        np.array(loss_std), np.array(mean_loss) + loss_std,
        alpha=.2, label=f"{label} variance over {npoints_to_average} steps")
