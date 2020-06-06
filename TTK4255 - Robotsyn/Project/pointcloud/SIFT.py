import numpy as np
import cv2
from matplotlib import pyplot as plt

def keypointAndDescriptor(img, nfeatures=0, mask = None):
    """
    Get list of keypoints and descriptors from an image
    Args:
        img:        Image (cv2.imread())
        nfeatures:  Number of features to use in image
    Returns:
        kp:         List of keypoints
        des:        List of descriptors
    """
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
    kp, des = sift.detectAndCompute(img, mask)

    return kp, des

def flannMatches2D2D(query_kp, train_kp, query_des, train_des):
    """
    Get list of point matches between two images
    Args:
        query_kp:    Keypoints from query image
        train_kp:    Keypoints from training image
        query_des:   Descriptors from query image
        train_des:   Descriptors from training image
    Returns:
        query_pts:      np.array of points in query image
        train_pts:      np.array of corresponding points in training image
        matches:        Raw array of matches from FLANN
        matches_mask:   Mask of good matches by Lowe's ratio test
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(query_des, train_des, k=2)

    # List of point matches
    query_pts = []
    train_pts = []

    # Need to draw only good matches, so create a mask
    matches_mask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matches_mask[i]=[1,0]

            # Append points
            query_pts.append(query_kp[m.queryIdx].pt)
            train_pts.append(train_kp[m.trainIdx].pt)

    return np.asarray(query_pts), np.asarray(train_pts), matches, matches_mask

def flannMatches3D2D(query_kp, query_des, train_model, train_des):
    """
    Get list of point matches between a 3D model and 2D image
    Args:
        query_kp:      Keypoints from query image
        query_des:     Descriptors from query image
        train_model:   np.array (nx3) of 3D points (X,Y,Z) in pointcloud
        train_des:     np.array (nxm) of corresponding descriptors for each point in pointcloud
    Returns:
        query_pts:      np.array of points in 3D model
        train_pts:      np.array of corresponding points in 2D image
        matches:        Raw array of matches from FLANN
        matches_mask:   Mask of good matches by Lowe's ratio test
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(query_des, train_des, k=2)

    # List of point matches
    query_pts = []
    train_pts = []

    # Need to draw only good matches, so create a mask
    matches_mask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matches_mask[i]=[1,0]

            # Append points
            query_pts.append(query_kp[m.queryIdx].pt)
            train_pts.append(train_model[m.trainIdx,:])

    return np.asarray(query_pts), np.asarray(train_pts), matches, matches_mask
 

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