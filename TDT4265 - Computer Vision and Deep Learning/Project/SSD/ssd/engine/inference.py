import logging
import torch
import torch.utils.data
import pathlib
from tqdm import tqdm
from ssd.data.build import make_data_loader
from ssd.data.datasets.evaluation import evaluate
from ssd import torch_utils


def convert_predictions(predictions):
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


@torch.no_grad()
def compute_on_dataset(model, data_loader):
    results_dict = {}
    for batch in tqdm(data_loader):
        images, targets, image_ids = batch
        images = torch_utils.to_cuda(images)
        outputs = model(images)

        outputs = [o.cpu() for o in outputs]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, outputs)}
        )
    return results_dict


def inference(model, data_loader, dataset_name, output_folder: pathlib.Path, **kwargs):
    dataset = data_loader.dataset
    logger = logging.getLogger("SSD.inference")
    logger.info(
        "Evaluating {} dataset({} images):".format(dataset_name, len(dataset)))
    predictions = compute_on_dataset(model, data_loader)
    predictions = convert_predictions(predictions)
    return evaluate(dataset=dataset, predictions=predictions, output_dir=output_folder, **kwargs)


@torch.no_grad()
def do_evaluation(cfg, model, **kwargs):
    model.eval()
    data_loaders_val = make_data_loader(cfg, is_train=False)
    eval_results = []
    for dataset_name, data_loader in zip(cfg.DATASETS.TEST, data_loaders_val):
        output_folder = pathlib.Path(cfg.OUTPUT_DIR, "inference", dataset_name)
        output_folder.mkdir(exist_ok=True, parents=True)
        eval_result = inference(
            model, data_loader, dataset_name, output_folder, **kwargs)
        eval_results.append(eval_result)
    return eval_results
