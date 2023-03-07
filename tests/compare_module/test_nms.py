import numpy as np
import torch as t
from compare_module.nms import nms


def test_nms_single_box():
    boxes = t.tensor([[10.0, 15.0, 25.0, 60.0],
                      [11.0, 13.0, 25.0, 55.0],
                      [10.0, 20.0, 40.0, 65.0],
                      [20.0, 5.0, 27.0, 60.0], ])
    scores = t.tensor([0.9, 0.7, 0.5, 0.1])
    labels = t.tensor([1, 1, 1, 1])
    nms_res, _ = nms(boxes, labels, scores, 0.3, func=None)
    test_boxes = t.tensor([[10.0, 15.0, 25.0, 60.0]])
    assert t.equal(nms_res, test_boxes)


def test_nms_multiple_boxes():
    boxes = t.tensor([[10.0, 15.0, 25.0, 60.0],
                      [11.0, 13.0, 25.0, 55.0],
                      [10.0, 20.0, 40.0, 65.0],
                      [90.0, 80.0, 180.0, 150.0], ])
    scores = t.tensor([0.9, 0.55, 0.5, 0.6])
    labels = t.tensor([1, 1, 1, 1])
    nms_res, _ = nms(boxes, labels, scores, 0.3, func=None)
    test_boxes = t.tensor([[10.0, 15.0, 25.0, 60.0],
                           [90.0, 80.0, 180.0, 150.0]])
    assert t.equal(nms_res, test_boxes)


def test_nms_no_matching_boxes():
    boxes = t.tensor([[10.0, 15.0, 25.0, 60.0],
                      [26.0, 65.0, 60.0, 100.0],
                      [10.0, 100.0, 40.0, 150.0],
                      [90.0, 80.0, 180.0, 150.0], ])
    scores = t.tensor([0.9, 0.55, 0.5, 0.6])
    labels = t.tensor([1, 1, 1, 1])
    nms_res, _ = nms(boxes, labels, scores, 0.3, func=None)
    test_boxes = t.tensor([[10.0, 15.0, 25.0, 60.0],
                           [90.0, 80.0, 180.0, 150.0],
                           [26.0, 65.0, 60.0, 100.0],
                           [10.0, 100.0, 40.0, 150.0], ])
    assert t.equal(nms_res, test_boxes)


def test_mean_same_classes():
    boxes = t.tensor([[10.0, 15.0, 25.0, 60.0],
                      [11.0, 13.0, 25.0, 55.0],
                      [10.0, 20.0, 40.0, 65.0],
                      [20.0, 5.0, 27.0, 60.0], ])
    scores = t.tensor([0.9, 0.7, 0.5, 0.1])
    labels = t.tensor([1, 1, 1, 1])
    mean_res, _ = nms(boxes, labels, scores, 0.3)
    test_boxes = t.tensor([[10.77, 15.05, 28.5, 59.55]])
    assert t.equal(mean_res, test_boxes)


def test_mean_diff_classes():
    boxes = t.tensor([[10.0, 15.0, 25.0, 60.0],
                      [11.0, 13.0, 25.0, 55.0],
                      [10.0, 20.0, 40.0, 65.0],
                      [20.0, 5.0, 27.0, 60.0], ])
    scores = t.tensor([0.9, 0.7, 0.5, 0.1])
    labels = t.tensor([1, 5, 1, 1])
    mean_res, _ = nms(boxes, labels, scores, 0.3)
    test_boxes = t.tensor([[10.67, 16.0, 30.13, 61.67]])
    assert t.equal(mean_res, test_boxes)
