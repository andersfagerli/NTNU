import argparse
import logging
import time
import torch
from ssd.config.defaults import cfg
from ssd.modeling.detector import SSDDetector
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.logger import setup_logger
from ssd import torch_utils
import torch.utils.data
from ssd.data.build import make_data_loader


@torch.no_grad()
def evaluation(cfg, ckpt, N_images: int):
    model = SSDDetector(cfg)
    logger = logging.getLogger("SSD.inference")
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR, logger=logger)
    model = torch_utils.to_cuda(model)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    model.eval()
    data_loaders_val = make_data_loader(cfg, is_train=False)
    for data_loader in  data_loaders_val:
        batch = next(iter(data_loader))
        images, targets, image_ids = batch
        images = torch_utils.to_cuda(images)
        imshape = list(images.shape[2:])
        # warmup
        print("Checking runtime for image shape:", imshape)
        for i in range(10):
            model(images)
        start_time = time.time()
        for i in range(N_images):
            outputs = model(images)
        total_time = time.time() - start_time
        print("Runtime for image shape:", imshape)
        print("Total runtime:", total_time)
        print("FPS:", N_images / total_time)


def main():
    parser = argparse.ArgumentParser(description='SSD Evaluation on VOC and COCO dataset.')
    parser.add_argument(
        "config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
        type=str,
    )
    parser.add_argument("--N_images", default=100, type=int, help="The number of images to check runtime with.")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TEST.BATCH_SIZE = 1
    cfg.freeze()

    logger = setup_logger("SSD", cfg.OUTPUT_DIR)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    evaluation(cfg, ckpt=args.ckpt, N_images=args.N_images)


if __name__ == '__main__':
    main()

