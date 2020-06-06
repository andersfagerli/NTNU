import sys
sys.path.append("../pointcloud")

# External libraries
import numpy as np

def readDescriptors(data_path, image_naming, image_format, num_images):
    """
    Reads descriptor values from .txt file exported to COLMAP, and returns them in original format
    Args:
        data_path:      Path of folder with images, e.g if images are in folder 'data/', the data path is 'data/'
        image_naming:   Labeling of images, e.g if images are labeled 'out1.png', the image naming is 'out'
        image_format:   Format of images, e.g if images are labeled 'out1.png', the image format is 'png'
        num_images:     Number of images to read
    Returns:
        descriptors:    Dictionary of descriptors in original OpenCV format
    """
    descriptors = {}
    for image_id in range(1, num_images + 1):
        if (image_id < 10):
            filename = data_path + image_naming + '00' + str(image_id) + '.' + image_format + '.txt'
        elif (image_id < 100):
            filename = data_path + image_naming + '0' + str(image_id) + '.' + image_format + '.txt'
        else:
            filename = data_path + image_naming + str(image_id) + '.' + image_format + '.txt'
        with open(filename, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) < 20: # Read header (NUM_FEATURES, 128)
                    elems = line.split()
                    num_features = int(elems[0])
                    descriptors_img = np.empty((num_features, 128))
                    curr_feature = 0
                if len(line) > 20 and curr_feature < num_features: # Read data (X, Y, SCALE, ORIENTATION, D_1, ..., D_128)
                    elems = line.split()
                    descriptors_img[curr_feature,:] = np.array([float(i) for i in elems[4:]])
                    curr_feature += 1

        descriptors[image_id] = descriptors_img
    return descriptors