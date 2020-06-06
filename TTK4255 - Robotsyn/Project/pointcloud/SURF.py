import numpy as np
import cv2
from matplotlib import pyplot as plt

def keypointAndDescriptor(img, threshold, mask=None):
    """
    Get list of keypoints and descriptors from an image
    Args:
        img:        Image (cv2.imread())
        nfeatures:  Number of features to use in image
    Returns:
        kp:         List of keypoints
        des:        List of descriptors
    """
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=threshold)
    surf.setExtended(True)
    kp, des = surf.detectAndCompute(img, mask)

    return kp, des

def flannMatches(kp1, kp2, des1, des2):
    """
    Get list of point matches between two images
    Args:
        kp1:    Keypoints from image 1
        kp2:    Keypoints from image 2
        des1:   Descriptors from image 1
        des2:   Descriptors from image 2
    Returns:
        pts1:           np.array of points in image 1
        pts2:           np.array of corresponding points in image 2
        matches:        Raw array of matches from FLANN
        matches_mask:   Mask of good matches by Lowe's ratio test
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # List of point matches
    pts1 = []
    pts2 = []

    # Need to draw only good matches, so create a mask
    matches_mask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matches_mask[i]=[1,0]

            # Append points
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    return np.asarray(pts1), np.asarray(pts2), matches, matches_mask
 
def drawMatches(img1, img2, kp1, kp2, matches, matches_mask):
    """
    Draws a pyplot of matches between two images
    Args:
        img1:           Image 1 (cv2.imread())
        img2:           Image 2 (cv2.imread())
        kp1:            Keypoints from image 1
        kp2:            Keypoints from image 2
        matches:        List of matches (output from flann.knnMatch())
        matches_mask:   Mask of matches from Lowe's ratio test
    """
    draw_params = dict(matchColor = (0,255,0),
                singlePointColor = (255,0,0),
                matchesMask = matches_mask,
                flags = 0)
    img = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img),plt.show()