"""
THERE SHOULD BE NO NEED TO EDIT THIS FILE.
"""

import json
import numpy as np
import os


def read_json_file(filepath):
    assert os.path.isfile(filepath), "Did not find filepath. \
                     I looked in: {}".format(os.path.abspath(filepath))
    with open(filepath, "r") as to_read:
        bounding_boxes = json.load(to_read)
    return bounding_boxes


def read_predicted_boxes():
    json_file = read_json_file("predicted_boxes.json")
    for image_id in json_file.keys():
        scores = np.array(json_file[image_id]["scores"])
        boxes = np.array(json_file[image_id]["boxes"])
        assert scores.shape[0] == boxes.shape[0]
        assert boxes.shape[1] == 4
        json_file[image_id]["scores"] = scores 
        json_file[image_id]["boxes"] = boxes
    return json_file


def read_ground_truth_boxes():
    json_file = read_json_file("ground_truth_boxes.json")
    for image_id in json_file.keys():
        boxes = np.array(json_file[image_id])
        assert boxes.shape[1] == 4
        json_file[image_id] = boxes
    return json_file