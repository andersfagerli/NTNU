import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # Intersection box top left
    # x1 = max(prediction_box[0], gt_box[0])
    # y1 = max(prediction_box[1], gt_box[1])
    # # Intersection box bottom right
    # x2 = min(prediction_box[2], gt_box[2])
    # y2 = min(prediction_box[3], gt_box[3])

    # if x1 > x2 or y1 > y2:
    #     iou = 0.0
    # else:
    #     # Box areas
    #     prediction_box_area = (prediction_box[2]-prediction_box[0]) * (prediction_box[3]-prediction_box[1])
    #     gt_box_area = (gt_box[2]-gt_box[0]) * (gt_box[3]-gt_box[1])
    #     # Compute intersection
    #     intersection = (x2 - x1) * (y2 - y1)
    #     # Compute union
    #     union = prediction_box_area + gt_box_area - intersection
    #     iou = intersection / union
    # assert iou >= 0 and iou <= 1
    # return iou
    leftX = max(prediction_box[0], gt_box[0])
    rightX = min(prediction_box[2], gt_box[2])
    topY = max(prediction_box[1], gt_box[1])
    bottomY = min(prediction_box[3], gt_box[3])

    intersectionArea = max(0, rightX - leftX) * max(0, bottomY - topY)
    # Compute union
    prediction_area = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    iou = intersectionArea / float(prediction_area + gt_area - intersectionArea)

    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp + num_fp == 0:
        return 1
    else:
        return num_tp / (num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp + num_fn == 0:
        return 0
    else:
        return num_tp / (num_tp + num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    matches = []
    for i in range(prediction_boxes.shape[0]):
        for j in range(gt_boxes.shape[0]):
            iou = calculate_iou(prediction_boxes[i,:], gt_boxes[j,:])
            if iou >= iou_threshold:
                matches.append([j,i,iou])

    # Sort all matches on IoU in descending order
    def get_iou(match):
        return match[2]

    def get_gt(match):
        return match[0]

    matches.sort(key=get_iou, reverse=True)
    matches.sort(key=get_gt)

    # Find all matches with the highest IoU threshold
    prediction_boxes_matched = []
    gt_boxes_matched = []
    prev_gt_idx = gt_boxes.shape[0]+1
    for i in range(len(matches)):
        current_gt_idx = matches[i][0]
        if current_gt_idx is not prev_gt_idx:
            prediction_boxes_matched.append(prediction_boxes[matches[i][1]])
            gt_boxes_matched.append(gt_boxes[matches[i][0]])

            prev_gt_idx = current_gt_idx

    return np.array(prediction_boxes_matched), np.array(gt_boxes_matched)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    TPFP = prediction_boxes.shape[0]

    prediction_boxes_matched, gt_boxes_matched = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

    TP = prediction_boxes_matched.shape[0]
    FP = TPFP - TP

    FN = gt_boxes.shape[0] - prediction_boxes_matched.shape[0]

    return {"true_pos": TP, "false_pos": FP, "false_neg": FN}


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    TP, FP, FN = 0, 0, 0
    for prediction_boxes, gt_boxes in zip(all_prediction_boxes, all_gt_boxes):
        image_result_dict = calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold)

        TP += image_result_dict["true_pos"]
        FP += image_result_dict["false_pos"]
        FN += image_result_dict["false_neg"]
    
    precision = calculate_precision(TP, FP, FN)
    recall = calculate_recall(TP, FP, FN)
    
    return (precision, recall)

def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # # YOUR CODE HERE
    precisions = [] 
    recalls = []
    for threshold in confidence_thresholds:
        accepted_prediction_boxes = []
        for image, prediction_boxes in enumerate(all_prediction_boxes):
            accepted_prediction_boxes.append(prediction_boxes[confidence_scores[image] >= threshold,:])
        
        precision, recall = calculate_precision_recall_all_images(accepted_prediction_boxes, all_gt_boxes, iou_threshold)

        precisions.append(precision)
        recalls.append(recall)
    
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    average_precision = 0
    for recall_level in recall_levels:
        max_precision = 0
        for i in range(precisions.shape[0]):
            if precisions[i] >= max_precision and recalls[i] >= recall_level:
                max_precision = precisions[i]
        
        average_precision += max_precision

    average_precision /= len(recall_levels)
    return average_precision

def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
