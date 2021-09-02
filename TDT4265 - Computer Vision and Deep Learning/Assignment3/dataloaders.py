from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import typing
import numpy as np
import PIL
np.random.seed(0)

mean = (0.5, 0.5, 0.5)
std = (.25, .25, .25)

def load_cifar10(
    batch_size: int,
    validation_fraction: float = 0.1,
    augment: bool = False,
    augment_extend: bool = False,
    size: int = -1,
    mean: tuple = mean,
    std: tuple = std) -> typing.List[torch.utils.data.DataLoader]:

    # Note that transform train will apply the same transform for
    # validation!
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize(size) if size is not -1 else transforms.Pad(0) # Pad(0) does nothing, only a placeholder
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize(size) if size is not -1 else transforms.Pad(0)
    ])
    transform_train_augmented = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.RandomCrop(32),
        transforms.Resize(size) if size is not -1 else transforms.Pad(0)
    ])
    data_train = datasets.CIFAR10('data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transform_train)

    data_test = datasets.CIFAR10('data/cifar10',
                                 train=False,
                                 download=True,
                                 transform=transform_test)
    
    if augment:
        data_train_augmented = datasets.CIFAR10('data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transform_train_augmented)
        if augment_extend:
            # Concatenate augmented dataset to original training data
            data_train = torch.utils.data.ConcatDataset((data_train, data_train_augmented))
        else:
            # Use augmented dataset as training data
            data_train = data_train_augmented
    
    indices = list(range(len(data_train)))
    split_idx = int(np.floor(validation_fraction * len(data_train)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(data_train,
                                                 sampler=validation_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=2)

    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

    return dataloader_train, dataloader_val, dataloader_test
