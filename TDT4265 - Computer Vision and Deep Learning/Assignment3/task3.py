import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
import numpy as np
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy

###################
### First model ###
###################

class ModelOne(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes,
                 batch_normalization: bool = False,
                 drop_out: bool = False,
                 conv_stride: bool = False,
                 init_weights: bool = False):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=image_channels, out_channels=num_filters, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=num_filters) if batch_normalization else nn.Identity(),
            nn.Conv2d(num_filters, num_filters, kernel_size=2, stride=2, padding=0) if conv_stride else nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout() if drop_out else nn.Identity(),
            # Layer 2
            nn.Conv2d(num_filters, num_filters*2, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters*2) if batch_normalization else nn.Identity(),
            nn.Conv2d(num_filters*2, num_filters*2, kernel_size=2, stride=2, padding=0) if conv_stride else nn.MaxPool2d(2, 2),
            nn.Dropout() if drop_out else nn.Identity(),
            # Layer 3
            nn.Conv2d(num_filters*2, num_filters*4, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters*4) if batch_normalization else nn.Identity(),
            nn.Conv2d(num_filters*4, num_filters*4, kernel_size=2, stride=2, padding=0) if conv_stride else nn.MaxPool2d(2, 2),
            nn.Dropout() if drop_out else nn.Identity(),
            # Flatten
            nn.Flatten()
        )
        self.num_output_features = num_filters*4*4*4
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64) if batch_normalization else nn.Identity(),
            nn.Linear(64, num_classes),
        )

        if init_weights:
            self.feature_extractor.apply(self.initialize_weights)
            self.classifier.apply(self.initialize_weights)

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        out = self.classifier(self.feature_extractor(x))
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out
    
    def initialize_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain('relu'))
            m.bias.data.fill_(0.01)


####################
### Second model ###
####################

class ModelTwo(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes,
                 batch_normalization: bool = False,
                 drop_out: bool = False,
                 conv_stride: bool = False,
                 init_weights: bool = False):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes

        conv_kernel_size = 5
        conv_padding_size = 2

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=image_channels, out_channels=num_filters, kernel_size=conv_kernel_size, stride=1, padding=conv_padding_size),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=num_filters) if batch_normalization else nn.Identity(),

            nn.Conv2d(num_filters, num_filters, conv_kernel_size, 1, conv_padding_size),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters) if batch_normalization else nn.Identity(),

            nn.Conv2d(num_filters, num_filters, 2, 2, 0) if conv_stride else nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5) if drop_out else nn.Identity(),
            # Layer 2
            nn.Conv2d(num_filters, num_filters*2, conv_kernel_size, 1, conv_padding_size),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters*2) if batch_normalization else nn.Identity(),

            nn.Conv2d(num_filters*2, num_filters*2, conv_kernel_size, 1, conv_padding_size),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters*2) if batch_normalization else nn.Identity(),

            nn.Conv2d(num_filters*2, num_filters*2, 2, 2, 0) if conv_stride else nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.4) if drop_out else nn.Identity(),
            # Layer 3
            nn.Conv2d(num_filters*2, num_filters*4, conv_kernel_size, 1, conv_padding_size),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters*4) if batch_normalization else nn.Identity(),

            nn.Conv2d(num_filters*4, num_filters*4, conv_kernel_size, 1, conv_padding_size),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters*4) if batch_normalization else nn.Identity(),

            nn.Conv2d(num_filters*4, num_filters*4, 2, 2, 0) if conv_stride else nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.3) if drop_out else nn.Identity(),
            # Flatten
            nn.Flatten()
        )

        self.num_output_features = num_filters*4*4*4
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64) if batch_normalization else nn.Identity(),
            nn.Linear(64, num_classes),
        )

        if init_weights:
            self.feature_extractor.apply(self.initialize_weights)
            self.classifier.apply(self.initialize_weights)

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        out = self.classifier(self.feature_extractor(x))
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out
    
    def initialize_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('relu'))
            m.bias.data.fill_(0.01)

def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64

    ###################
    ### First model ###
    ###################

    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size, augment=False, augment_extend=False)
    model = ModelOne(
        image_channels=3,
        num_classes=10,
        batch_normalization=True,
        drop_out=False,
        conv_stride=False,
        init_weights=False
    )
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders,
        l2_reg=0,
        lr_schedule_gamma=0.1,
        use_adam=False
    )
    trainer.train()

    create_plots(trainer, "task2")

    # Best model accuracies and loss
    trainer.load_best_model()
    dataloader_train, dataloader_val, dataloader_test = dataloaders

    train_loss, train_accuracy = compute_loss_and_accuracy(dataloader_train, trainer.model, nn.CrossEntropyLoss())
    val_loss, val_accuracy = compute_loss_and_accuracy(dataloader_val, trainer.model, nn.CrossEntropyLoss())
    test_loss, test_accuracy = compute_loss_and_accuracy(dataloader_test, trainer.model, nn.CrossEntropyLoss())

    print("\nAccuracies and loss of first model")
    print("Training accuracy:\t", train_accuracy, "\t\tTraining loss:\t", train_loss)
    print("Validation accuracy:\t", val_accuracy, "\t\tValidation loss:", val_loss)
    print("Test accuracy:\t\t", test_accuracy, "\t\tTest loss:\t", test_loss)

    ####################
    ### Second model ###
    ####################

    learning_rate = 0.1
    early_stop_count = 20
    dataloaders = load_cifar10(batch_size, augment=False, augment_extend=False)
    model = ModelTwo(
        image_channels=3,
        num_classes=10,
        batch_normalization=True,
        drop_out=False,
        conv_stride=False,
        init_weights=False
    )
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders,
        l2_reg=0,
        lr_schedule_gamma=0.25,
        use_adam=False
    )
    trainer.train()

    create_plots(trainer, "task2")

    # Best model accuracies
    trainer.load_best_model()
    dataloader_train, dataloader_val, dataloader_test = dataloaders

    train_loss, train_accuracy = compute_loss_and_accuracy(dataloader_train, trainer.model, nn.CrossEntropyLoss())
    val_loss, val_accuracy = compute_loss_and_accuracy(dataloader_val, trainer.model, nn.CrossEntropyLoss())
    test_loss, test_accuracy = compute_loss_and_accuracy(dataloader_test, trainer.model, nn.CrossEntropyLoss())

    print("\nAccuracies and loss of second model")
    print("Training accuracy:\t", train_accuracy, "\t\tTraining loss:\t", train_loss)
    print("Validation accuracy:\t", val_accuracy, "\t\tValidation loss:", val_loss)
    print("Test accuracy:\t\t", test_accuracy, "\t\tTest loss:\t", test_loss)

    
    ### Task 3d ###
    # Deactivate batch normalization

    learning_rate = 0.1
    early_stop_count = 20
    dataloaders = load_cifar10(batch_size, augment=False, augment_extend=False)
    model = ModelTwo(
        image_channels=3,
        num_classes=10,
        batch_normalization=False,
        drop_out=False,
        conv_stride=False,
        init_weights=False
    )
    trainerTest = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders,
        l2_reg=0,
        lr_schedule_gamma=0.25,
        use_adam=False
    )
    trainerTest.train()

    plt.figure(figsize=(20, 8))
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss - With method", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss - With method")
    utils.plot_loss(trainerTest.train_history["loss"], label="Training loss - Without method", npoints_to_average=10)
    utils.plot_loss(trainerTest.validation_history["loss"], label="Validation loss - Without method")
    plt.legend()
    plt.show()

    # Best model accuracies and loss without batch normalization
    trainerTest.load_best_model()
    dataloader_train, dataloader_val, dataloader_test = dataloaders

    train_loss, train_accuracy = compute_loss_and_accuracy(dataloader_train, trainerTest.model, nn.CrossEntropyLoss())
    val_loss, val_accuracy = compute_loss_and_accuracy(dataloader_val, trainerTest.model, nn.CrossEntropyLoss())
    test_loss, test_accuracy = compute_loss_and_accuracy(dataloader_test, trainerTest.model, nn.CrossEntropyLoss())

    print("\nAccuracies and loss of test model")
    print("Training accuracy:\t", train_accuracy, "\t\tTraining loss:\t", train_loss)
    print("Validation accuracy:\t", val_accuracy, "\t\tValidation loss:", val_loss)
    print("Test accuracy:\t\t", test_accuracy, "\t\tTest loss:\t", test_loss)