import numpy as np
import pathlib
import matplotlib.pyplot as plt
import matplotlib
from vizer.draw import draw_boxes
import torch

from train import get_parser
from ssd.config.defaults import cfg
from ssd.data.build import make_data_loader
from ssd.modeling.box_head.prior_box import PriorBox
from ssd.utils.box_utils import convert_locations_to_boxes
from statistics import build_train_dataset_and_loader

np.random.seed(0)

def visualize_training_set(cfg, image_id = "Czech_000006"):
    data_loader = make_data_loader(cfg, is_train=True)
    if isinstance(data_loader, list):
        data_loader = data_loader[0]
    dataset = data_loader.dataset

    image = dataset._read_image(image_id)
    boxes, labels = dataset._get_annotation(image_id)
    image = draw_boxes(
        image, boxes, labels, class_name_map=dataset.class_names
    )

    plt.imshow(image)
    plt.show()

def visualize_validation_set(cfg, amount=200):
    output_dir = pathlib.Path('visualizations/validation_set')
    output_dir.mkdir(exist_ok=True, parents=True)

    data_loader = make_data_loader(cfg, is_train=False)
    if isinstance(data_loader, list):
        data_loader = data_loader[0]
    dataset = data_loader.dataset
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    print("Generating images ..")
    counter = 0
    for idx in indices:
        image_id = dataset.image_ids[idx]
        image = dataset._read_image(image_id)
        boxes, labels = dataset._get_annotation(image_id)
        #image = draw_boxes(
        #    image, boxes, labels, class_name_map=dataset.class_names
        #)
        
        plt.imsave("visualizations/validation_set/visualization"+str(idx)+".png", image)

        counter += 1

        if counter >= amount:
            break
    
    print("Saved the images to visualizations/")

def visualize_prior_boxes(cfg, layer):
    def plot_bbox(ax, box, color, do_plot=True, circle=True):
        cx, cy, w, h = box
        cx *= cfg.INPUT.IMAGE_SIZE[0]
        cy *= cfg.INPUT.IMAGE_SIZE[1]
        w *= cfg.INPUT.IMAGE_SIZE[0]
        h *= cfg.INPUT.IMAGE_SIZE[1]
        x1, y1 = cx + w/2, cy + h/2
        x0, y0 = cx - w/2, cy - h/2
        if do_plot:
            if circle:
                ax.add_artist(matplotlib.patches.Ellipse([cx, cy], w,h, alpha=.1, color=color))
            else:
                ax.add_artist(plt.Rectangle([x0, y0], x1-x0, y1-y0, alpha=0.3, color=color))
                #plt.plot([x0, x0, x1, x1, x0],[y0, y1, y1, y0, y0], f"{color}", alpha=.5, color=color)
            plt.plot(cx, cy, f"o{color}")
        else:
            plt.plot(cx, cy, f"o{color}", alpha=0.1)

    
    def get_num_boxes_in_fmap(idx):
        boxes_per_location = cfg.MODEL.PRIORS.BOXES_PER_LOCATION[idx]
        feature_map_size = cfg.MODEL.PRIORS.FEATURE_MAPS[idx]
        return int(boxes_per_location * np.prod(feature_map_size))

    PLOT_CIRCLE = False
    # Set which priors we want to visualize
    # 0 is the last layer
    layer_to_visualize = layer
    # Set which aspect ratio indices we want to visualize
    aspect_ratio_indices = list(range(cfg.MODEL.PRIORS.BOXES_PER_LOCATION[layer_to_visualize]))

    fig, ax = plt.subplots()
    # Create prior box
    prior_box = PriorBox(cfg)
    priors = prior_box()
    print("Prior box shape:", priors.shape)
    # Prior boxes are saved such that all prior boxes at the first feature map is saved first, then all prios at the next (lower) feature map
    print("First prior example:", priors[5])
    locations = torch.zeros_like(priors)[None]
    priors_as_location = convert_locations_to_boxes(locations, priors,cfg.MODEL.CENTER_VARIANCE, cfg.MODEL.SIZE_VARIANCE)[0]

    # Set up our scene
    plt.ylim([-100, cfg.INPUT.IMAGE_SIZE[1]+100])
    plt.xlim([-100, cfg.INPUT.IMAGE_SIZE[0]+100])

    offset = sum([get_num_boxes_in_fmap(prev_layer) for prev_layer in range(layer_to_visualize)])
    boxes_per_location = cfg.MODEL.PRIORS.BOXES_PER_LOCATION[layer_to_visualize]
    indices_to_visualize = []
    colors = []
    available_colors = ["r", "g", "b", "y", "m", "b","w"]
    for idx in range(offset, offset + get_num_boxes_in_fmap(layer_to_visualize)):
        for aspect_ratio_idx in aspect_ratio_indices:
            if idx % boxes_per_location == aspect_ratio_idx:
                indices_to_visualize.append(idx)
                colors.append(available_colors[aspect_ratio_idx])
    ax.add_artist(plt.Rectangle([0, 0], cfg.INPUT.IMAGE_SIZE[0], cfg.INPUT.IMAGE_SIZE[1], fill=False, edgecolor="black"))
    do_plot = False

    # Only plot prior boxes at middle, for visability
    indice_to_plot = len(indices_to_visualize)/2

    for i, idx in enumerate(indices_to_visualize):
        prior = priors_as_location[idx]
        color = colors[i]
        if i >= (indice_to_plot - len(aspect_ratio_indices)//2) and i < (indice_to_plot + len(aspect_ratio_indices)//2):
            do_plot = True
        else:
            do_plot = False

        plot_bbox(ax, prior, color, do_plot, PLOT_CIRCLE)
    plt.show()


if __name__ == "__main__":
    config_path = "configs/train_rdd2020.yaml"
    cfg.merge_from_file(config_path)
    cfg.freeze()

    visualize_validation_set(cfg)
    #visualize_training_set(cfg)
    
    #visualize_prior_boxes(cfg, layer=3)
    