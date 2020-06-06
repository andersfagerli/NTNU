import numpy as np

def essential_from_fundamental(F, K1, K2):
    """ Computes the essential matrix from F.

    See HZ Ch. 9.6 (p257): Formula 9.12.

    Args:
        F:  Fundamental Matrix
        K1: Intrinsic matrix for camera 1
        K2: Intrinsic matrix for camera 2

    Returns:
        E:  Essential Matrix
    """

    return K1.T @ F @ K2
