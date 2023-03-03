import numpy as np
from compare_module.iou import compute_IoU


def test_compute_IoU():
    _test_compute_IoU(np.array([[10., 10., 15., 30.]]),
                      np.array([[12., 15., 30., 50.]]),
                      0.066)
    _test_compute_IoU(np.array([[10., 10., 30., 20.]]),
                      np.array([[16., 13., 24., 16.]]),
                      0.12)
    _test_compute_IoU(np.array([[10., 10., 15., 30.]]),
                      np.array([[10., 10., 15., 30.]]),
                      1.0)
    _test_compute_IoU(np.array([[10., 10., 15., 30.]]),
                      np.array([[35., 10., 55., 30.]]),
                      0.0)


def _test_compute_IoU(box_1, box_2, exp_val):
    assert round(compute_IoU(box_1, box_2)[0], 3) == exp_val
