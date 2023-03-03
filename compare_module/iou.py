import numpy as np


def compute_IoU(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0])
    ih = np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua
