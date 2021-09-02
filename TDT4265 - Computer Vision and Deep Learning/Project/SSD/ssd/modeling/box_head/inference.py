import torch
import torchvision
from ssd.container import Container


class PostProcessor:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.width = cfg.INPUT.IMAGE_SIZE[1]
        self.height = cfg.INPUT.IMAGE_SIZE[0]

    def __call__(self, detections):
        """
        Post processor of detections when we are doing inference.
        Applies two filter operations to remove all prior boxes that does not include an object:
        Confidence thresholding and non-maximum suppression.
        """
        batches_scores, batches_boxes = detections
        device = batches_scores.device
        batch_size = batches_scores.size(0)
        results = []
        for batch_id in range(batch_size):
            scores, boxes = batches_scores[batch_id], batches_boxes[batch_id]  # (N, #CLS) (N, 4)
            num_boxes = scores.shape[0]
            num_classes = scores.shape[1]

            boxes = boxes.view(num_boxes, 1, 4).expand(num_boxes, num_classes, 4)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, num_classes).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            indices = torch.nonzero(scores > self.cfg.TEST.CONFIDENCE_THRESHOLD).squeeze(1)
            boxes, scores, labels = boxes[indices], scores[indices], labels[indices]

            boxes[:, 0::2] *= self.width
            boxes[:, 1::2] *= self.height

            keep = batched_nms(boxes, scores, labels, self.cfg.TEST.NMS_THRESHOLD)
            # keep only topk scoring predictions
            keep = keep[:self.cfg.TEST.MAX_PER_IMAGE]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            container = Container(boxes=boxes, labels=labels, scores=scores)
            container.img_width = self.width
            container.img_height = self.height
            results.append(container)
        return results


def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = torchvision.ops.nms(boxes_for_nms, scores, iou_threshold)
    return keep
