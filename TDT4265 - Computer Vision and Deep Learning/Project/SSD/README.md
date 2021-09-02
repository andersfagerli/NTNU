# SSD300 In Pytorch 1.3
This implementation is a slimmed down version of: https://github.com/lufficc/SSD
Thanks to the original authors for creating the amazing repository and setting a MIT license on it.

### Features
This code base will be the start for your assignment 4 and the final project.
You can customize a ton of stuff with it, for example:

1. There exists a bunch of already implemented data augmentation techniques in [transforms.py](ssd/data/transforms/transform.py)
2. There is an extensive config file that you can set. For example, in the .yml files in configs/, we can override the defaults. To check out the default config, see: [defaults.py](ssd/config/defaults.py)
3. Two datasets are currently supported: MNIST Object detection, and PASCAL VOC.
4. Tensorboard support. Everything is logged to tensorboard, and you can check out the logs in either the [custom notebook](plot_scalars.ipynb), or launching a tensorboard with the command: `tensorboard --logdir outputs`

## Tutorials
- [Introduction to code. Training, evaluating and inference on demo images](tutorials/code_introduction.md)
- [Environment setup](tutorials/environment_setup.md)
- [Tensorboard logging](tutorials/tensorboard.md)
- [Dataset setup](tutorials/dataset.md)
- [Evaluation and submission of result (only required for project)](tutorials/evaluation_tdt4265.md)
