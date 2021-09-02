from yacs.config import CfgNode as CN

cfg = CN()

cfg.MODEL = CN()
# match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
cfg.MODEL.THRESHOLD = 0.5
cfg.MODEL.NUM_CLASSES = 21
# Hard negative mining
cfg.MODEL.NEG_POS_RATIO = 3
cfg.MODEL.CENTER_VARIANCE = 0.1
cfg.MODEL.SIZE_VARIANCE = 0.2

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
cfg.MODEL.BACKBONE = CN()
cfg.MODEL.BACKBONE.NAME = 'vgg'
cfg.MODEL.BACKBONE.OUT_CHANNELS = (512, 1024, 512, 256, 256, 256)
cfg.MODEL.BACKBONE.PRETRAINED = True
cfg.MODEL.BACKBONE.FREEZE = False
cfg.MODEL.BACKBONE.INPUT_CHANNELS = 3

# -----------------------------------------------------------------------------
# PRIORS
# -----------------------------------------------------------------------------
cfg.MODEL.PRIORS = CN()
# X, Y 
cfg.MODEL.PRIORS.FEATURE_MAPS = [[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]]
# X, Y
cfg.MODEL.PRIORS.STRIDES = [[8, 8], [16, 16], [32, 32], [64, 64], [100, 100], [300, 300]]
# X, Y
cfg.MODEL.PRIORS.MIN_SIZES = [[30, 30], [60, 60], [111, 111], [162, 162], [213, 213], [264, 264]]
# X, Y
cfg.MODEL.PRIORS.MAX_SIZES = [[60, 60], [111, 111], [162, 162], [213, 213], [264, 264], [315, 315]]
cfg.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# When has 1 aspect ratio, every location has 4 boxes, 2 ratio 6 boxes.
# #boxes = 2 + #ratio * 2
cfg.MODEL.PRIORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 4, 4]  # number of boxes per feature map location
cfg.MODEL.PRIORS.CLIP = True

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
cfg.INPUT = CN()
# Image size
cfg.INPUT.IMAGE_SIZE = [300, 300]
# Values to be used for image normalization, RGB layout
cfg.INPUT.PIXEL_MEAN = [123.675, 116.280, 103.530]
cfg.INPUT.PIXEL_STD = [1, 1, 1]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
cfg.DATASETS = CN()
# List of the dataset names for training, as present in pathscfgatalog.py
cfg.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in pathscfgatalog.py
cfg.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
cfg.DATA_LOADER = CN()
# Number of data loading threads
cfg.DATA_LOADER.NUM_WORKERS = 4
cfg.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Solver - The same as optimizer
# ---------------------------------------------------------------------------- #
cfg.SOLVER = CN()
# train configs
cfg.SOLVER.TYPE = "adam"
cfg.SOLVER.MAX_ITER = 120000
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.BATCH_SIZE = 32
cfg.SOLVER.LR = 1e-3
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.WEIGHT_DECAY = 5e-4
cfg.SOLVER.MULTISTEP_MILESTONES = [cfg.SOLVER.MAX_ITER]
cfg.SOLVER.WARMUP_PERIOD = 0

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
cfg.TEST = CN()
cfg.TEST.NMS_THRESHOLD = 0.45
cfg.TEST.CONFIDENCE_THRESHOLD = 0.01
cfg.TEST.MAX_PER_CLASS = -1
cfg.TEST.MAX_PER_IMAGE = 100
cfg.TEST.BATCH_SIZE = 10
cfg.EVAL_STEP = 500 # Evaluate dataset every eval_step, disabled when eval_step < 0
cfg.MODEL_SAVE_STEP = 500 # Save checkpoint every save_step
cfg.LOG_STEP = 10 # Print logs every log_stepPrint logs every log_step
cfg.OUTPUT_DIR = "outputs"
cfg.DATASET_DIR = "datasets"

cfg.PRETRAINED_PATH = ""
