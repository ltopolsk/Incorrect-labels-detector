import numpy as np
import torch as t


def compute_IoU(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = t.minimum(a[:, 2], b[:, 2]) - t.maximum(a[:, 0], b[:, 0])
    ih = t.minimum(a[:, 3], b[:, 3]) - t.maximum(a[:, 1], b[:, 1])

    iw = t.clamp(iw, min=0.0)
    ih = t.clamp(ih, min=0.0)

    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + area - iw * ih

    ua = t.clamp(ua, min=t.finfo(t.float32).eps)

    intersection = iw * ih

    return intersection / ua


if __name__ == '__main__':
    box_1 = t.tensor([[10., 10., 15., 30.]])
    box_2 = t.tensor([[12., 15., 30., 50.]])
    print(round(compute_IoU(box_1, box_2).item(), 3))
