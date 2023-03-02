import numpy as np
import compare_module.compare as cmp


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
    assert round(cmp.compute_IoU(box_1, box_2)[0], 3) == exp_val


def test_compare():
    boxes = cmp.compare(np.array([]),
                        np.array([]),
                        np.array([[15., 20., 60., 100],
                                  [50., 10., 100., 30]]),
                        np.array([1, 10]))

    exp_dicts = [{'box_mean': None,
                  'label_mean': None,
                  'box_test': [15., 20., 60., 100],
                  'label_test': 1,
                  'err': cmp.ERR_BBOX_UNNES, },
                 {'box_mean': None,
                  'label_mean': None,
                  'box_test': [50., 10., 100., 30],
                  'label_test': 10,
                  'err': cmp.ERR_BBOX_UNNES, }]
    for item1, item2 in zip(boxes, exp_dicts):
        assert item1 == item2

    boxes = cmp.compare(np.array([[15., 20., 60., 100],
                                  [50., 10., 100., 30]]),
                        np.array([1, 10]),
                        np.array([]),
                        np.array([]))

    exp_dicts = [{'box_mean': [15., 20., 60., 100],
                  'label_mean': 1,
                  'box_test': None,
                  'label_test': None,
                  'err': cmp.ERR_LACK_BBOX, },
                 {'box_mean': [50., 10., 100., 30],
                  'label_mean': 10,
                  'box_test': None,
                  'label_test': None,
                  'err': cmp.ERR_LACK_BBOX, }]
    for item1, item2 in zip(boxes, exp_dicts):
        assert item1 == item2

    boxes = cmp.compare(np.array([[15., 20., 60., 100],
                                  [50., 10., 100., 30]]),
                        np.array([1, 10]),
                        np.array([[15., 20., 60., 100],
                                  [50., 10., 100., 30]]),
                        np.array([10, 9]))

    exp_dicts = [{'box_mean': [15., 20., 60., 100],
                  'label_mean': 1,
                  'box_test': [15., 20., 60., 100],
                  'label_test': 10,
                  'err': cmp.ERR_LABEL, },
                 {'box_mean': [50., 10., 100., 30],
                  'label_mean': 10,
                  'box_test': [50., 10., 100., 30],
                  'label_test': 9,
                  'err': cmp.ERR_LABEL, }]
    for item1, item2 in zip(boxes, exp_dicts):
        assert item1 == item2

    boxes = cmp.compare(np.array([[15., 20., 60., 100],
                                  [50., 10., 100., 30]]),
                        np.array([1, 10]),
                        np.array([[15., 20., 60., 100],
                                  [50., 10., 100., 30]]),
                        np.array([1, 10]))

    exp_dicts = [{'box_mean': [15., 20., 60., 100],
                  'label_mean': 1,
                  'box_test': [15., 20., 60., 100],
                  'label_test': 1,
                  'err': cmp.TRUE_POS, },
                 {'box_mean': [50., 10., 100., 30],
                  'label_mean': 10,
                  'box_test': [50., 10., 100., 30],
                  'label_test': 10,
                  'err': cmp.TRUE_POS, }]
    for item1, item2 in zip(boxes, exp_dicts):
        assert item1 == item2

    boxes = cmp.compare(np.array([[50., 10., 100., 30]]),
                        np.array([10]),
                        np.array([[15., 20., 60., 100],
                                  [50., 10., 100., 30]]),
                        np.array([1, 10]))

    exp_dicts = [{'box_mean': [50., 10., 100., 30],
                  'label_mean': 10,
                  'box_test': [50., 10., 100., 30],
                  'label_test': 10,
                  'err': cmp.TRUE_POS, },
                 {'box_mean': None,
                  'label_mean': None,
                  'box_test': [15., 20., 60., 100],
                  'label_test': 1,
                  'err': cmp.ERR_BBOX_UNNES, }, ]
    for item1, item2 in zip(boxes, exp_dicts):
        assert item1 == item2

    boxes = cmp.compare(np.array([[15., 10., 300., 150],
                                  [50., 10., 55., 15]]),
                        np.array([1, 10]),
                        np.array([[15., 20., 60., 100],
                                  [50., 10., 100., 30]]),
                        np.array([1, 10]))

    exp_dicts = [{'box_mean': [15., 10., 300., 150],
                  'label_mean': 1,
                  'box_test': [15., 20., 60., 100],
                  'label_test': 1,
                  'err': cmp.ERR_BBOX_SIZE, },
                 {'box_mean': [50., 10., 55., 15],
                  'label_mean': 10,
                  'box_test': [50., 10., 100., 30],
                  'label_test': 10,
                  'err': cmp.ERR_BBOX_SIZE, }]
    for item1, item2 in zip(boxes, exp_dicts):
        assert item1 == item2

    boxes = cmp.compare(np.array([[15., 20., 60., 100],
                                  [15., 20., 60., 100]]),
                        np.array([1, 10]),
                        np.array([[15., 20., 60., 100]]),
                        np.array([1]))

    exp_dicts = [{'box_mean': [15., 20., 60., 100],
                  'label_mean': 1,
                  'box_test': [15., 20., 60., 100],
                  'label_test': 1,
                  'err': cmp.TRUE_POS, },
                 {'box_mean':  [15., 20., 60., 100],
                  'label_mean': 10,
                  'box_test': None,
                  'label_test': None,
                  'err': cmp.ERR_LACK_BBOX, }]
    for item1, item2 in zip(boxes, exp_dicts):
        assert item1 == item2
