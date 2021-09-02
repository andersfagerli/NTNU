import torchvision
from torch import nn
import matplotlib.pyplot as plt

import utils
from trainer import Trainer, compute_loss_and_accuracy
from dataloaders import load_cifar10

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10) # No need to apply softmax, as this is done in nn.CrossEntropyLoss

        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected
            param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
            param.requires_grad = True # layers
    
    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    utils.set_seed(0)
    epochs = 10
    batch_size = 32

    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(
        batch_size,
        augment=False,
        augment_extend=False,
        size=224,
        mean=(0.485, 0.456, 0.406),
        std= (0.229, 0.224, 0.225))

    model = Model()

    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders,
        l2_reg=0,
        lr_schedule_gamma=0.0,
        use_adam=True
    )
    trainer.train()

    plt.figure(figsize=(20, 8))
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.show()

    # Best model accuracies and loss
    trainer.load_best_model()
    dataloader_train, dataloader_val, dataloader_test = dataloaders

    train_loss, train_accuracy = compute_loss_and_accuracy(dataloader_train, trainer.model, nn.CrossEntropyLoss())
    val_loss, val_accuracy = compute_loss_and_accuracy(dataloader_val, trainer.model, nn.CrossEntropyLoss())
    test_loss, test_accuracy = compute_loss_and_accuracy(dataloader_test, trainer.model, nn.CrossEntropyLoss())

    print("\nAccuracies and loss of best model")
    print("Training accuracy:\t", train_accuracy, "\t\tTraining loss:\t", train_loss)
    print("Validation accuracy:\t", val_accuracy, "\t\tValidation loss:", val_loss)
    print("Test accuracy:\t\t", test_accuracy, "\t\tTest loss:\t", test_loss)