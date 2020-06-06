# https://colmap.github.io/tutorial.html#feature-detection-and-extraction

import sys
sys.path.append("../pointcloud")

# External libraries
import numpy as np
import cv2

# Own libraries
import SIFT as sift
import SURF as surf
import util

# TODO: Generalize to other descriptors (SURF, ..)
def export_to_colmap(data_path, image_naming, image_format, num_images):
    """
    Exports features detected by a descriptor to COLMAP import format as .txt files
    Args:
        data_path:      Path of folder with images, e.g if images are in folder 'data/', the data path is 'data/'
        image_naming:   Labeling of images, e.g if images are labeled 'out1.png', the image naming is 'out'
        image_format:   Format of images, e.g if images are labeled 'out1.png', the image format is 'png'
        num_images:     Number of images to perform feature detection on (assumes images are numbered as 1,2,3, ..., e.g out1, out2, ..)
    """
    nfeatures = 0  # SIFT param: maximum number of features to detect in each image ("0" to give all detected features)
    hessian_threshold = 400 # SURF param

    ### Main loop ###
    for i in range(1, num_images + 1):
        if (i < 10):
            filename = data_path + image_naming + '00' + str(i) + '.' + image_format
        elif (i < 100):
            filename = data_path + image_naming + '0' + str(i) + '.' + image_format
        else:
            filename = data_path + image_naming + str(i) + '.' + image_format

        img = cv2.imread(filename, 0)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        start_y = int(img.shape[0]/10)
        start_x = int(img.shape[1]/4)
        cv2.rectangle(mask, (start_x, start_y), (img.shape[1] - start_x, img.shape[0] - start_y), (255), thickness=-1)
        
        kp, des = surf.keypointAndDescriptor(img, hessian_threshold, mask)
        print(des.shape)
        num_features = len(kp)
        
        file = open(filename + '.txt', "w")

        file.write(str(num_features) + ' ' + str(128) + '\n')

        for j in range(num_features):
            x = kp[j].pt[0]
            y = kp[j].pt[1]
            scale = kp[j].size
            orientation = kp[j].angle
            descriptor = des[j,:]

            file.write(' '.join(map(str, [x,y,scale,orientation])) + ' ')
            file.write(' '.join(map(str, descriptor)))

            if (j != (num_features-1)): # Write new lines except for last line
                file.write('\n')

        file.close()
    
if __name__ == "__main__":
    data_path = 'data/'
    image_naming = 'out'
    image_format = 'png'
    num_images = 30

    export_to_colmap(data_path, image_naming, image_format, num_images)