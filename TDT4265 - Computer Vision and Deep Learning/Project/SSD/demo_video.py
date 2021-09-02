import pathlib
import tqdm
import cv2
from ssd.config.defaults import cfg
import argparse
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import moviepy.editor as mp
import tempfile
from demo import run_demo


def dump_frames(video, directory: pathlib.Path):
    for frame_idx, frame in enumerate(
            tqdm.tqdm(video.iter_frames(), desc="Reading video frames")):
        impath = pathlib.Path(directory, f"{frame_idx}.png")
        cv2.imwrite(str(impath), frame[:, :, ::-1])


def infer_video(
        cfg, ckpt, video_path: str, score_threshold: float,
        dataset_type, output_path):
    assert pathlib.Path(video_path).is_file(),\
        f"Did not find video: {video_path}"
    with tempfile.TemporaryDirectory() as cache_dir:
        input_image_dir = pathlib.Path(cache_dir, "input_images")
        input_image_dir.mkdir()
        with mp.VideoFileClip(video_path) as video:
            original_fps = video.fps
            dump_frames(video, input_image_dir)

        output_image_dir = pathlib.Path(cache_dir, "video_images")
        output_image_dir.mkdir(exist_ok=True)
        run_demo(
            cfg, ckpt,
            score_threshold,
            pathlib.Path(input_image_dir),
            output_image_dir,
            dataset_type)

        impaths = list(output_image_dir.glob("*.png"))
        impaths.sort(key=lambda impath: int(impath.stem))
        impaths = [str(impath) for impath in impaths]
        with mp.ImageSequenceClip(impaths, fps=original_fps) as video:
            video.write_videofile(output_path)


def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "video_path", type=str, metavar="FILE",
        help="Path to source video")
    parser.add_argument(
        "output_path", type=str,
        help="Output path to save video with detections")
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument("--dataset_type", default="tdt4265", type=str, help='Specify dataset type. Currently support voc and coco.')

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    infer_video(cfg=cfg,
                ckpt=args.ckpt,
                score_threshold=args.score_threshold,
                video_path=args.video_path,
                output_path=args.output_path,
                dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()
