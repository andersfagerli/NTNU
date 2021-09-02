
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]

# Forward image through all convolutional layers
modules = list(model.children())
for i in range(len(modules)-2):
    image = modules[i].forward(image)

def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        image: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image


indices = list(range(10))
fig, axs = plt.subplots(2, len(indices)//2)

print("Activation shape of last layer: ", image.shape)

for i, indice in enumerate(indices):
    activation = torch_image_to_numpy(image[0,indice])
    if i < 5:
        axs[0,i].imshow(activation, cmap="gray")
    else:
        axs[1,i-5].imshow(activation, cmap="gray")

plt.show()
