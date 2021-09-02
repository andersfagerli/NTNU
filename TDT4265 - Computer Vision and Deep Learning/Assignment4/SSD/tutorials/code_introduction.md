# Introduction to code

## Train
Train for:
1. MNIST:
```bash
python train.py  configs/mnist.yaml
```
2. MNIST on the tdt4265.idi.ntnu.no server (This only changes the "datasets" variable in the config file):
```bash
python train.py  configs/mnist_tdt4265_server.yaml
```
3. VGG SSD300 on the PACAL VOC dataset
```bash
python train.py  configs/vgg_ssd300_voc0712.yaml
```

## Evaluate
Run test.py to evaluate the whole validation dataset: 
```bash
python test.py configs/mnist.yaml
```
Remember to give the correct config path

## Demo
For Pascal VOC
```bash
python demo.py configs/vgg_ssd300_voc0712.yaml --images_dir demo/voc --dataset_type voc
```

For MNIST Object detection
```bash
python demo.py  configs/mnist.yaml --images_dir demo/voc --dataset_type mnist
```

You will see a similar output:
```text
(0001/0005) 004101.jpg: objects 01 | load 010ms | inference 033ms | FPS 31
(0002/0005) 003123.jpg: objects 05 | load 009ms | inference 019ms | FPS 53
(0003/0005) 000342.jpg: objects 02 | load 009ms | inference 019ms | FPS 51
(0004/0005) 008591.jpg: objects 02 | load 008ms | inference 020ms | FPS 50
(0005/0005) 000542.jpg: objects 01 | load 011ms | inference 019ms | FPS 53
```


## Deeper dive
For assignment 4, you don't need to understand much about the repository. Just change the files as guided in assignment 4, and you should be fine.
For the object detection task in the project, this introduction is to give you a better understanding.

The code is structured in three sub-modules:
```
- config
- data
- engine
- modeling
- utils
```

The following will give you a small introduction to the different folders:

### config
The config folder includes path_catalog, which is of no interest as this should work out of the box, and the default config settings.
Here you can take a look at what you can change, and this might improve your results.
However, you should **not** change the settings in `ssd/config/defaults`. Instead, you should *override* the config.
This can be done in `configs/name_of_config_file.yml`.

### data
Data includes everything of loading of datasets, evaluation for each dataset, and possible data-augmentation transforms.
For you, the transforms are interesting. This can be defined in `ssd/data/transforms/__init__.py`. We have not included any data-augmenting transforms in the starter code, but you can take a look at the pytorch docs or what is available in `ssd/data/transforms/transforms.py`.

### engine
Includes inference.py and trainer.py
trainer.py is responsible for the gradient descent train loop, and engine.py for evaluation code.

### modeling
This is the most relevant for your project and assignment 4.
Your probably don't need to change `ssd/modeling/box_head`, as it's responsible for implementing the SSD "head", where `ssd/modeling/box_head/prior_box.py` generates prior boxes.
`ssd/modeling/box_head/backbone` contains the backbones (currently only VGG and the basic model for assignment 4). These are built in the function `build_backbone` in `ssd/modeling/detector.py`.

### utils
Nothing special here, but different utilities files to make it all work together.
