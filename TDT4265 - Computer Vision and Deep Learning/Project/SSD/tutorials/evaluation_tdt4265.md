# Evaluation of models on the TDT4265 Dataset

To run evaluation on the dataset, we've created an automatic python script for you:

Run `submit_results.py` with the config file to your model, for example:
```bash
python3 submit_results.py configs/train_tdt4265.yml
```

This will load the most recent checkpoint and perform inference on the entire tdt4265 test dataset.

Note that the mAP result you get is not the result on the entire dataset (only 30%). The mAP for the entire dataset will be public after the project submission deadline.

## Evaluation on video

Run `demo_video.py` with the config file to your model, the path to the source video, and an output path. For example,
```bash
python3 demo_video.py configs/train_tdt4265.yml /some/path/to/source/video.mp4 output.mp4
```

This will load the most recent checkpoint and perform inference on the video.

If you're working on a TDT4265 computer, we've already downloaded the videos for you. You can run:
```bash
python3 demo_video.py configs/train_tdt4265.yaml /work/datasets/2019-12-05_18-26-20-front_split2.mp4 output1.mp4
python3 demo_video.py configs/train_tdt4265.yaml /work/datasets/2019-12-06_09-44-38-front_split1.mp4 output2.mp4
```