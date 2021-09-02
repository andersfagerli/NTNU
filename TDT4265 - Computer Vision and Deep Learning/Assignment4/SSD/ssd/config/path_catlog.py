import os


class DatasetCatalog:
    DATA_DIR = ''
    DATASETS = {
        'voc_2007_train': {
            "data_dir": "VOCdevkit/VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "VOCdevkit/VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "VOCdevkit/VOC2007",
            "split": "trainval"
        },
        'voc_2007_test': {
            "data_dir": "VOCdevkit/VOC2007",
            "split": "test"
        },
        'voc_2012_train': {
            "data_dir": "VOCdevkit/VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "VOCdevkit/VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "VOCdevkit/VOC2012",
            "split": "trainval"
        },
        'voc_2012_test': {
            "data_dir": "VOCdevkit/VOC2012",
            "split": "test"
        },
        'coco_2014_valminusminival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_valminusminival2014.json"
        },
        'coco_2014_minival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_minival2014.json"
        },
        'coco_2014_train': {
            "data_dir": "train2014",
            "ann_file": "annotations/instances_train2014.json"
        },
        'coco_2014_val': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_val2014.json"
        },
        'mnist_detection_train': {
            'data_dir': 'mnist_detection/train',
            'split': 'train'
        },
        'mnist_detection_val': {
            'data_dir': 'mnist_detection/test',
            'split': 'val'
        },
        'waymo_train': {
            'data_dir': 'waymo',
            'split': "train"
        },
        'waymo_val': {
            'data_dir': 'waymo',
            'split': 'val'
        },
        'tdt4265_train': {
            'data_dir': 'tdt4265/train',
            'split': 'train'
        },
        'tdt4265_val': {
            'data_dir': 'tdt4265/train',
            'split': 'val'
        },
        'tdt4265_test': {
            'data_dir': 'tdt4265/test',
            'split': 'test'
        },
        "rdd2020_train": {
            "data_dir": "RDD2020_filtered",
            "split": "train"
        },
        "rdd2020_val": {
            "data_dir": "RDD2020_filtered",
            "split": "train"
        }

    }

    @staticmethod
    def get(base_path, name):
        assert name in DatasetCatalog.DATASETS,\
            f"Did not find dataset: {name} in dataset catalog. {DatasetCatalog.DATASETS.keys()}"
        root = os.path.join(base_path, DatasetCatalog.DATA_DIR)
        attrs = DatasetCatalog.DATASETS[name]
        data_dir = os.path.join(root, DatasetCatalog.DATA_DIR, attrs["data_dir"])
        if "voc" in name:
            args = dict(data_dir=data_dir, split=attrs["split"])
            return dict(factory="VOCDataset", args=args)
        elif "coco" in name:
            args = dict(
                data_dir=data_dir,
                ann_file=os.path.join(root, attrs["ann_file"]),
            )
            return dict(factory="COCODataset", args=args)
        elif "mnist_detection" in name:
            args = dict(
                data_dir=data_dir, is_train=attrs["split"] == "train")
            return dict(factory="MNISTDetection", args=args)
        elif "waymo" in name:
            args = dict(data_dir=data_dir, split=attrs["split"])
            return dict(factory="WaymoDataset", args=args)
        elif "rdd2020" in name:
            args = dict(data_dir=data_dir, split=attrs["split"])
            return dict(factory="RDDDataset", args=args)
        elif "tdt4265" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(data_dir=data_dir, split=attrs["split"]
                )
            return dict(factory="TDT4265Dataset", args=args)
        raise RuntimeError("Dataset not available: {}".format(name))
