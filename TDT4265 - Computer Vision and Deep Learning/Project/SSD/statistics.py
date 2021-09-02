import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from vizer.draw import draw_boxes

from ssd.config.defaults import cfg
from ssd.data.build import BatchCollator, make_data_loader
from ssd.data.datasets import build_dataset
from ssd.data import samplers
from ssd.data.transforms.transforms import *
from ssd.modeling.box_head.prior_box import PriorBox
from ssd.data.transforms.target_transform import SSDTargetTransform

def build_train_transforms(cfg):
    
    transform = [
        ConvertFromInts(),
        Resize(cfg.INPUT.IMAGE_SIZE),
        ToTensor(),
    ]
    
    transform = Compose(transform)
    return transform

def build_train_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform

def build_train_dataset_and_loader(cfg):
    train_transform = build_train_transforms(cfg)
    target_transform = build_train_target_transform(cfg)
    datasets = build_dataset(
        base_path=cfg.DATASET_DIR,
        dataset_list=cfg.DATASETS.TRAIN,
        transform=train_transform,
        target_transform=target_transform)
    dataset = datasets[0]
    
    sampler = torch.utils.data.RandomSampler(dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=cfg.SOLVER.BATCH_SIZE, drop_last=True)
    loader = DataLoader(dataset, num_workers=cfg.DATA_LOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                        pin_memory=cfg.DATA_LOADER.PIN_MEMORY, collate_fn=BatchCollator(True))
    
    return dataset, loader

def calculate_mean_and_std(dataset, loader):
    # Calculate mean
    total_sum_rgb = torch.tensor([0.0, 0.0, 0.0])
    total_pixels = len(dataset) * cfg.INPUT.IMAGE_SIZE[0] * cfg.INPUT.IMAGE_SIZE[1]
    for (images, _, _) in loader:
        total_sum_rgb += torch.sum(images, (0,2,3))
        
    means = total_sum_rgb / total_pixels
    
    # Calculate std
    sum_of_squared_error = torch.tensor([0.0, 0.0, 0.0])
    for (images, _, _) in loader:
        for image in images:
            sum_of_squared_error[0] += torch.sum((image[0]-means[0]).pow(2))
            sum_of_squared_error[1] += torch.sum((image[1]-means[1]).pow(2))
            sum_of_squared_error[2] += torch.sum((image[2]-means[2]).pow(2))

    stds = torch.sqrt(sum_of_squared_error / total_pixels)

    return means, stds

def get_distribution_of_data(dataset):
    amount_of_labels =  {1: 0, 2: 0, 3: 0, 4: 0}
    aspect_ratios =     {1: [], 2: [], 3: [], 4: []}
    areas =             {1: [], 2: [], 3: [], 4: []}

    for idx in range(len(dataset)):
        image_id = dataset.image_ids[idx]
        boxes, labels = dataset._get_annotation(image_id)
        for label, box in zip(labels, boxes):
            amount_of_labels[label] += 1

            width = box[2] - box[0]
            height = box[3] - box[1]
            if width < 1.0 or height < 1.0:
                continue

            aspect_ratios[label].append(height/width)
            areas[label].append(height*width)

    amount_of_labels = {"D00": amount_of_labels[1], "D10": amount_of_labels[2], "D20": amount_of_labels[3], "D40": amount_of_labels[4]}
    aspect_ratios = {"D00": aspect_ratios[1], "D10": aspect_ratios[2], "D20": aspect_ratios[3], "D40": aspect_ratios[4]}
    areas = {"D00": areas[1], "D10": areas[2], "D20": areas[3], "D40": areas[4]}  

    return amount_of_labels, aspect_ratios, areas

def get_outliers(data_dict):
    outlier_dict = {}
    key_num = 1
    for (key, val) in data_dict.items():
        box = plt.boxplot(val)
        whiskers = box["whiskers"]

        outlier_limits = []
        
        outlier_limits.append(np.min(whiskers[0].get_ydata()))
        outlier_limits.append(np.max(whiskers[1].get_ydata()))

        outlier_dict[key_num] = outlier_limits
        key_num +=1
    return outlier_dict

def get_aspect_outlier_image_ids(dataset, outliers):
    image_ids = []
    for idx in range(len(dataset)):
        image_id = dataset.image_ids[idx]
        boxes, labels = dataset._get_annotation(image_id)
        for label, box in zip(labels, boxes):
            width = box[2] - box[0]
            height = box[3] - box[1]
            if width < 1.0 or height < 1.0:
                error = np.inf
                image_ids.append((image_id, error))
                continue

            aspect_ratio = height/width
            if aspect_ratio < outliers[label][0]:
                error = np.abs(aspect_ratio-outliers[label][0])
                image_ids.append((image_id, error))
            elif aspect_ratio > outliers[label][1]:
                error = np.abs(aspect_ratio-outliers[label][1])
                image_ids.append((image_id, error))
    
    def get_error(x):
        return x[1]

    image_ids.sort(key=get_error, reverse=True)
    image_ids = [i[0] for i in image_ids]
    image_ids = list(dict.fromkeys(image_ids))

    return image_ids
    

def get_area_outlier_image_ids(dataset, outliers):
    image_ids = []
    for idx in range(len(dataset)):
        image_id = dataset.image_ids[idx]
        boxes, labels = dataset._get_annotation(image_id)
        for label, box in zip(labels, boxes):
            width = box[2] - box[0]
            height = box[3] - box[1]

            area = height*width
            if area < outliers[label][0]:
                error = np.abs(area-outliers[label][0])
                image_ids.append((image_id, error))
            elif area > outliers[label][1]:
                error = np.abs(area-outliers[label][1])
                image_ids.append((image_id, error))
    
    def get_error(x):
        return x[1]

    image_ids.sort(key=get_error, reverse=True)
    image_ids = [i[0] for i in image_ids]
    image_ids = list(dict.fromkeys(image_ids))

    return image_ids

def visualize_outlier_images(dataset):
    train_labels, train_aspects, train_areas = get_distribution_of_data(train_dataset)
    validation_labels, validation_aspects, validation_areas = get_distribution_of_data(validation_dataset)

    aspect_outliers = get_outliers(train_aspects)
    area_outliers = get_outliers(train_areas)

    area_image_ids = get_area_outlier_image_ids(train_dataset, area_outliers)
    aspect_image_ids = get_aspect_outlier_image_ids(train_dataset, aspect_outliers)

    output_dir = pathlib.Path('visualizations/outliers/area')
    output_dir.mkdir(exist_ok=True, parents=True)

    output_dir = pathlib.Path('visualizations/outliers/aspectratio')
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Generating images ..")
    counter = 0
    for i in reversed(range(len(area_image_ids))):
        image_id = area_image_ids[i]
        image = dataset._read_image(image_id)
        boxes, labels = dataset._get_annotation(image_id)
        image = draw_boxes(
            image, boxes, labels, class_name_map=dataset.class_names
        )
        
        plt.imsave("visualizations/outliers/area/"+str(image_id)+".png", image)

        counter += 1

        if counter >= 100:
            break

    counter = 0
    for image_id in aspect_image_ids:
        image = dataset._read_image(image_id)
        boxes, labels = dataset._get_annotation(image_id)
        image = draw_boxes(
            image, boxes, labels, class_name_map=dataset.class_names
        )
        
        plt.imsave("visualizations/outliers/aspectratio/"+str(image_id)+".png", image)

        counter += 1

        if counter >= 100:
            break
    
    print("Saved the images to visualizations/outliers")

def plot_boxplot(dataset):
    labels, aspects, areas = get_distribution_of_data(dataset)
    
    # Aspect ratios
    fig, ax = plt.subplots(1,4)
    ax[0].boxplot(aspects["D00"], meanline=True, showmeans=True, showfliers=False)
    ax[0].xaxis.set_visible(False)
    ax[0].set_title("D00")

    ax[1].boxplot(aspects["D10"], meanline=True, showmeans=True, showfliers=False)
    ax[1].xaxis.set_visible(False)
    ax[1].set_title("D10")

    ax[2].boxplot(aspects["D20"], meanline=True, showmeans=True, showfliers=False)
    ax[2].xaxis.set_visible(False)
    ax[2].set_title("D20")

    ax[3].boxplot(aspects["D40"], meanline=True, showmeans=True, showfliers=False)
    ax[3].xaxis.set_visible(False)
    ax[3].set_title("D40")

    fig.suptitle("Aspect ratios - height/width")
    plt.show()

    # Areas
    fig, ax = plt.subplots(1,4)
    ax[0].boxplot(areas["D00"], meanline=True, showmeans=True, showfliers=False)
    ax[0].xaxis.set_visible(False)
    ax[0].set_title("D00")

    ax[1].boxplot(areas["D10"], meanline=True, showmeans=True, showfliers=False)
    ax[1].xaxis.set_visible(False)
    ax[1].set_title("D10")

    ax[2].boxplot(areas["D20"], meanline=True, showmeans=True, showfliers=False)
    ax[2].xaxis.set_visible(False)
    ax[2].set_title("D20")

    ax[3].boxplot(areas["D40"], meanline=True, showmeans=True, showfliers=False)
    ax[3].xaxis.set_visible(False)
    ax[3].set_title("D40")

    fig.suptitle("Areas - height*width")
    plt.show()


if __name__ == "__main__":
    cfg.merge_from_file("configs/train_tdt4265.yaml")
    cfg.freeze()

    train_dataset, train_loader = build_train_dataset_and_loader(cfg)

    validation_loader = make_data_loader(cfg, is_train=False)
    if isinstance(validation_loader, list):
        validation_loader = validation_loader[0]
    validation_dataset = validation_loader.dataset

    #means, stds = calculate_mean_and_std(train_dataset, train_loader)

    #print("Mean: ", means)
    #print("Std: ", stds)
    #print("Std/255: ", stds/255)

    #visualize_outlier_images(train_dataset)

    plot_boxplot(train_dataset)
    