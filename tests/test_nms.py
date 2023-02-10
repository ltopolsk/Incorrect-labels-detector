import numpy as np
from compare_module.compare import nms


def test_nms():
    boxes = np.array([[10.0, 15.0, 25.0, 60.0],
                      [11.0, 13.0, 25.0, 55.0],
                      [10.0, 20.0, 40.0, 65.0],
                      [20.0, 5.0, 27.0, 60.0], ])
    scores = np.array([0.9, 0.7, 0.5, 0.1])
    nms_res = nms(boxes, scores, 0.3)
    # print(nms_res)
    test_boxes = [np.array([10.0, 15.0, 25.0, 60.0])]
    assert len(nms_res) == len(test_boxes)
    for nms_box, test_box in zip(nms_res, test_boxes):
        assert np.array_equal(nms_box, test_box)