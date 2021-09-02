import pathlib
import tqdm
import torch
import json
import time
from PIL import Image
from ssd.config.defaults import cfg
from ssd.config.path_catlog import DatasetCatalog
from ssd.data.datasets import TDT4265Dataset
import argparse
import numpy as np
from ssd import torch_utils
from ssd.data.transforms import build_transforms
from ssd.modeling.detector import SSDDetector
from ssd.utils.checkpoint import CheckPointer


LABEL_MAP = {
    'D00 - Linear Longitudinal Crack': 0,
    'D10 - Linear Lateral Crack': 1,
    'D20 - Alligator and Other Complex Cracks': 2,
    'D40 - Pothole': 3
}

@torch.no_grad()
def get_detections(cfg, ckpt):
    model = SSDDetector(cfg)
    model = torch_utils.to_cuda(model)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    dataset_path = DatasetCatalog.DATASETS["tdt4265_test"]["data_dir"]
    dataset_path = pathlib.Path(cfg.DATASET_DIR, dataset_path)
    image_dir = pathlib.Path(dataset_path)
    image_paths = list(image_dir.glob("*.jpg"))

    transforms = build_transforms(cfg, is_train=False)
    model.eval()
    detections = []
    for image_path in tqdm.tqdm(image_paths, desc="Inference on images"):
        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        result = model(torch_utils.to_cuda(images))[0]
        result = result.resize((width, height)).cpu().numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']
        for idx in range(len(boxes)):
            box = boxes[idx]
            label_id = labels[idx]
            label = TDT4265Dataset.class_names[label_id]
            assert label != "__background__"
            score = float(scores[idx])
            assert box.shape == (4,)
            xmin, ymin, xmax, ymax = box.tolist()
            width = xmax - xmin
            height = ymax - ymin
            detections.append(
                {
                    "image_id": int(image_path.stem),
                    "category_id": LABEL_MAP[label],
                    "score": score,
                    "bbox": [xmin, ymin, width, height]
                }
            )
    return detections


def dump_detections(cfg, detections, path):
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as fp:
        json.dump(detections, fp)
    print("Detections saved to:", path)
    print("Abolsute path:", path.absolute())
    print("Go to: https://tdt4265-annotering.idi.ntnu.no/submissions/ to submit your result")


def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    detections = get_detections(
        cfg=cfg,
        ckpt=args.ckpt)
    json_path = pathlib.Path(cfg.OUTPUT_DIR, "test_detected_boxes.json")
    dump_detections(cfg, detections, json_path)


if __name__ == '__main__':
    main()
