import torch
from torch import nn
import torch.nn.functional as F
from ssd.modeling.box_head.prior_box import PriorBox
from ssd.utils import box_utils
from .inference import PostProcessor
from .loss import MultiBoxLoss


class SSDBoxHead(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = BoxPredictor(cfg)
        self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.post_processor = PostProcessor(cfg)
        self.priors = None

    def forward(self, features, targets=None):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred, targets)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred, targets):
        gt_boxes, gt_labels = targets["boxes"], targets["labels"]
        reg_loss, cls_loss = self.loss_evaluator(
            cls_logits, bbox_pred, gt_labels, gt_boxes
        )
        loss_dict = dict(reg_loss=reg_loss, cls_loss=cls_loss,)
        detections = (cls_logits, bbox_pred)
        return detections, loss_dict

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred,
            self.priors,
            self.cfg.MODEL.CENTER_VARIANCE,
            self.cfg.MODEL.SIZE_VARIANCE,
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        detections = self.post_processor(detections)
        return detections, {}


class BoxPredictor(nn.Module):
    """
    The class responsible for generating predictions for each prior
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()
        for level, (boxes_per_location, out_channels) in enumerate(
            zip(cfg.MODEL.PRIORS.BOXES_PER_LOCATION, cfg.MODEL.BACKBONE.OUT_CHANNELS)
        ):
            self.cls_headers.append(
                self.cls_block(level, out_channels, boxes_per_location)
            )
            self.reg_headers.append(
                self.reg_block(level, out_channels, boxes_per_location)
            )
        self.init_parameters()

    def cls_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(
            out_channels,
            boxes_per_location * self.cfg.MODEL.NUM_CLASSES,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def reg_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(
            out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1
        )

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        cls_logits = []
        bbox_preds = []
        for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
            cls_logit = cls_header(feature).permute(0, 2, 3, 1).contiguous()
            bbox_pred = reg_header(feature).permute(0, 2, 3, 1).contiguous()
            cls_logits.append(cls_logit)
            bbox_preds.append(bbox_pred)

        batch_size = features[0].shape[0]
        cls_logits = torch.cat(
            [c.view(c.shape[0], -1) for c in cls_logits],
            dim=1).view(batch_size, -1, self.cfg.MODEL.NUM_CLASSES)
        bbox_preds = torch.cat(
            [l.view(l.shape[0], -1) for l in bbox_preds],
            dim=1).view(batch_size, -1, 4)

        return cls_logits, bbox_preds
