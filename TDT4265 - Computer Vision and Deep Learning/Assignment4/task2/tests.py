from task2 import *
import numpy as np


def test_iou():
    print("="*80)
    print("Running tests for calculate_iou_individual_image")
    b1 = np.array([0, 0, 1, 1])
    b2 = np.array([1.0, 1.0, 2, 2])

    res = calculate_iou(b1, b2)
    ans = 0
    assert res == ans, "Expected {}, got: {}".format(ans, res)
    b1 = np.array([2, 1, 4, 3])
    b2 = np.array([1, 2, 3, 4])

    res = calculate_iou(b1, b2)
    ans = 1/7
    assert res == ans, "Expected {}, got: {}".format(ans, res)

    b1 = np.array([0, 0, 1, 1])
    b2 = np.array([0, 0, 1, 1])
    res = calculate_iou(b1, b2)
    ans = 1.0
    assert res == ans, "Expected {}, got: {}".format(ans, res)

    b1 = np.array([0, 0, 1, 1])
    b2 = np.array([0.5, 0.5, 1, 1])
    res = calculate_iou(b1, b2)
    ans = 0.25
    assert res == ans, "Expected {}, got: {}".format(ans, res)

    b1 = np.array([5.5, 5.5, 8, 8])
    b2 = np.array([5.5, 3, 8, 4])
    res = calculate_iou(b1, b2)
    ans = 0.0
    assert res == ans, "Expected {}, got: {}".format(ans, res)

    b1 = np.array([5.5, 5.5, 8, 8])
    b2 = np.array([3, 5.5, 4, 9])
    res = calculate_iou(b1, b2)
    ans = 0.0
    assert res == ans, "Expected {}, got: {}".format(ans, res)

    b1 = np.array([522, 540, 576, 660])
    b2 = np.array([520, 540, 570, 655])
    res = round(calculate_iou(b1, b2), 5)
    ans = 0.82265
    assert res == ans, "Expected {}, got: {}".format(ans, res)


def test_precision():
    print("="*80)
    print("Running tests for calculate_precision")
    ans = 1
    res = calculate_precision(0, 0, 0)
    assert res == ans, "Expected {}, got: {}".format(ans, res)

    res = calculate_precision(10, 20, 0)
    ans = 1/3
    assert res == ans, "Expected {}, got: {}".format(ans, res)


def test_recall():
    print("="*80)
    print("Running tests for calculate_recall")
    ans = 0
    res = calculate_recall(0, 0, 0)
    assert res == ans, "Expected {}, got: {}".format(ans, res)

    res = calculate_recall(10, 0, 30)
    ans = 1/4
    assert res == ans, "Expected {}, got: {}".format(ans, res)


def test_get_all_box_matches():
    print("="*80)
    print("Running tests for get_all_box_matches")
    b1 = np.array([
        [0, 0, 1, 1]
    ])
    b2 = np.array([
        [0, 0, 1, 1]
    ])
    res1, res2 = get_all_box_matches(b1, b2, 0.5)
    assert np.all(res1 == b1)
    assert np.all(res2 == b2)
    res1, res2 = get_all_box_matches(b1, b2, 1)
    assert np.all(res1 == b1)
    assert np.all(res2 == b2)

    b2 = np.array([
        [0, 0, 1, 1],
        [0.25, 0.25, 1, 1]
    ])
    res1, res2 = get_all_box_matches(b1, b2, 1)
    assert np.all(res1 == b1)
    assert np.all(res2 == b2[0:1])

    b2 = np.array([
        [0.25, 0.25, 1, 1],
        [0, 0, 1, 1]
    ])
    res1, res2 = get_all_box_matches(b1, b2, 1)
    assert np.all(res1 == b1)
    assert np.all(res2 == b2[1:2])

    res1, res2 = get_all_box_matches(np.array([]), np.array([]), 0.5)
    assert res1.size == 0
    assert res2.size == 0


def test_calculate_individual_image_result():
    print("="*80)
    print("Running tests for calculate_individual_image_result")
    b1 = np.array([
        [0, 0, 1, 1],
        [0.5, 0.5, 1.5, 1.5],
        [2, 2, 3, 3],
        [5.5, 5.5, 8, 8]
    ])
    b2 = np.array([
        [0, 0, 1, 1],
        [0, 0, 1.5, 1.5],
        [3, 3, 4, 4],
        [5, 5, 8, 8]
    ])
    np.random.shuffle(b1)
    np.random.shuffle(b2)
    ans1 = 2
    ans2 = 2
    ans3 = 2
    res = calculate_individual_image_result(b1, b2, 0.5)

    assert res["true_pos"] == ans1, "Expected {}, got: {}".format(
        ans1, res["true_pos"])
    assert res["false_pos"] == ans2, "Expected {}, got: {}".format(
        ans2, res["false_pos"])
    assert res["false_neg"] == ans3, "Expected {}, got: {}".format(
        ans3, res["false_neg"])


def test_calculate_precision_recall_all_images():
    print("="*80)
    print("Running tests for calculate_precision_recall_all_images")
    b1 = np.array([
        [0, 0, 1, 1],
        [0.5, 0.5, 1.5, 1.5],
        [2, 2, 3, 3],
        [5.5, 5.5, 8, 8]
    ])
    b2 = np.array([
        [0, 0, 1, 1],
        [0, 0, 1.5, 1.5],
        [3, 3, 4, 4],
        [5, 5, 8, 8]
    ])
    np.random.shuffle(b1)
    np.random.shuffle(b2)
    ans1 = 6/8
    ans2 = 6/8
    res1, res2 = calculate_precision_recall_all_images([b1, b2], [b2, b2], 0.5)
    assert res1 == ans1, "Expected {}, got: {}".format(ans1, res1)
    assert res2 == ans2, "Expected {}, got: {}".format(ans2, res2)


def test_get_precision_recall_curve():
    print("="*80)
    print("Running tests for get_precision_recall_curve")
    b1 = np.array([
        [0, 0, 1, 1],
        [0.5, 0.5, 1.5, 1.5],
        [2, 2, 3, 3],
        [5.5, 5.5, 8, 8]
    ])
    b2 = np.array([
        [0, 0, 1, 1],
        [0, 0, 1.5, 1.5],
        [3, 3, 4, 4],
        [5, 5, 8, 8]
    ])
    s = np.array([0.4, 0.7, 0.6, 0.9])
    ans1 = 404
    ans2 = 243
    res1, res2 = get_precision_recall_curve([b1, b2], [b2, b2], [s, s], 0.5)
    res1 = int(res1.sum())
    res2 = int(res2.sum())
    assert res1 == ans1, "Expected {}, got: {}".format(ans1, res1)
    assert res2 == ans2, "Expected {}, got: {}".format(ans2, res2)


def test_mean_average_precision():
    print("="*80)
    print("Running tests for calculate_mean_average_precision")
    p = np.array([0.19620253, 0.38137083, 0.65555556, 0.81179423, 0.88598901,
                  0.93198263, 0.95386905, 0.9695586, 0.98397436, 1.])
    r = np.array([0.99237805, 0.99237805, 0.98932927, 0.98628049, 0.98323171,
                  0.98170732, 0.97713415, 0.97103659, 0.93597561, 0.])

    res1 = calculate_mean_average_precision(p, r)
    ans1 = 0.89598
    assert round(res1, 5) == ans1, "Expected {}, got: {}".format(ans1, res1)


if __name__ == "__main__":
    test_iou()
    test_precision()
    test_recall()
    test_get_all_box_matches()
    test_calculate_individual_image_result()
    test_calculate_precision_recall_all_images()
    test_get_precision_recall_curve()
    test_mean_average_precision()
    print("="*80)
    print("All tests OK.")