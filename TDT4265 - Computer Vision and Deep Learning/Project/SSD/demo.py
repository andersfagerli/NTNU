import pathlib
import torch
import tqdm
from PIL import Image
from vizer.draw import draw_boxes
from ssd.config.defaults import cfg
from ssd.data.datasets import VOCDataset, MNISTDetection, RDDDataset, TDT4265Dataset
import argparse
import numpy as np
from ssd import torch_utils
from ssd.data.transforms import build_transforms
from ssd.modeling.detector import SSDDetector
from ssd.utils.checkpoint import CheckPointer


@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir: pathlib.Path, output_dir: pathlib.Path, dataset_type):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == "mnist": 
        class_names = MNISTDetection.class_names 
    elif dataset_type == "tdt4265": 
        class_names = TDT4265Dataset.class_names
    elif dataset_type == "rdd2020":
        class_names = RDDDataset.class_names
    else:
        raise NotImplementedError('Not implemented now.')

    model = SSDDetector(cfg)
    model = torch_utils.to_cuda(model)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))

    output_dir.mkdir(exist_ok=True, parents=True)

    transforms = build_transforms(cfg, is_train=False)
    model.eval()
    drawn_images = []
    for i, image_path in enumerate(tqdm.tqdm(image_paths, desc="Predicting on images")):
        image_name = image_path.stem

        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)

        result = model(torch_utils.to_cuda(images))[0]

        result = result.resize((width, height)).cpu().numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']
        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        drawn_image = draw_boxes(
            image, boxes, labels, scores, class_names).astype(np.uint8)
        drawn_images.append(drawn_image)
        im = Image.fromarray(drawn_image)
        output_path = output_dir.joinpath(f"{image_name}.png")
        im.save(output_path)
    return drawn_images


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
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument("--images_dir", default='demo/voc', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--dataset_type", default="voc", type=str, help='Specify dataset type.')

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

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=pathlib.Path(args.images_dir),
             output_dir=pathlib.Path(args.images_dir, "result"),
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()
