import argparse
import logging
from ssd.config.defaults import cfg
from ssd.engine.inference import do_evaluation
from ssd.modeling.detector import SSDDetector
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.logger import setup_logger
from ssd import torch_utils


def evaluation(cfg, ckpt):
    logger = logging.getLogger("SSD.inference")

    model = SSDDetector(cfg)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR, logger=logger)
    model = torch_utils.to_cuda(model)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    do_evaluation(cfg, model)


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
    parser.add_argument("--output_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
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

    logger = setup_logger("SSD", cfg.OUTPUT_DIR)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    evaluation(cfg, ckpt=args.ckpt)


if __name__ == '__main__':
    main()
