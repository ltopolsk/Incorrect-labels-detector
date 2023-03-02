import numpy as np
from compare_module.compare import mean_bbox, nms


def test_nms():
    boxes = np.array([[10.0, 15.0, 25.0, 60.0],
                      [11.0, 13.0, 25.0, 55.0],
                      [10.0, 20.0, 40.0, 65.0],
                      [20.0, 5.0, 27.0, 60.0], ])
    scores = np.array([0.9, 0.7, 0.5, 0.1])
    labels = np.array([1, 1, 1, 1])
    nms_res, _ = nms(boxes, labels, scores, 0.3)
    test_boxes = np.array([[10.0, 15.0, 25.0, 60.0]])
    assert np.array_equal(nms_res, test_boxes)
    boxes = np.array([[10.0, 15.0, 25.0, 60.0],
                      [11.0, 13.0, 25.0, 55.0],
                      [10.0, 20.0, 40.0, 65.0],
                      [90.0, 80.0, 180.0, 150.0], ])
    scores = np.array([0.9, 0.55, 0.5, 0.6])
    labels = np.array([1, 1, 1, 1])
    nms_res, _ = nms(boxes, labels, scores, 0.3)
    test_boxes = np.array([[10.0, 15.0, 25.0, 60.0],
                           [90.0, 80.0, 180.0, 150.0]])
    assert np.array_equal(nms_res, test_boxes)
    boxes = np.array([[10.0, 15.0, 25.0, 60.0],
                      [26.0, 65.0, 60.0, 100.0],
                      [10.0, 100.0, 40.0, 150.0],
                      [90.0, 80.0, 180.0, 150.0], ])
    scores = np.array([0.9, 0.55, 0.5, 0.6])
    labels = np.array([1, 1, 1, 1])
    nms_res, _ = nms(boxes, labels, scores, 0.3)
    test_boxes = np.array([[10.0, 15.0, 25.0, 60.0],
                           [90.0, 80.0, 180.0, 150.0],
                           [26.0, 65.0, 60.0, 100.0],
                           [10.0, 100.0, 40.0, 150.0], ])
    assert np.array_equal(nms_res, test_boxes)


def test_mean_s():
    boxes = np.array([[10.0, 15.0, 25.0, 60.0],
                      [11.0, 13.0, 25.0, 55.0],
                      [10.0, 20.0, 40.0, 65.0],
                      [20.0, 5.0, 27.0, 60.0], ])
    scores = np.array([0.9, 0.7, 0.5, 0.1])
    labels = np.array([1, 1, 1, 1])
    mean_res, _ = nms(boxes, labels, scores, 0.3, func=mean_bbox)
    test_boxes = np.array([[10.77, 15.05, 28.5, 59.55]])
    assert np.array_equal(mean_res, test_boxes)
    boxes = np.array([[10.0, 15.0, 25.0, 60.0],
                      [11.0, 13.0, 25.0, 55.0],
                      [10.0, 20.0, 40.0, 65.0],
                      [20.0, 5.0, 27.0, 60.0], ])
    scores = np.array([0.9, 0.7, 0.5, 0.1])
    labels = np.array([1, 5, 1, 1])
    mean_res, _ = nms(boxes, labels, scores, 0.3, func=mean_bbox)
    test_boxes = np.array([[10.67, 16.0, 30.13, 61.67]])
    assert np.array_equal(mean_res, test_boxes)
