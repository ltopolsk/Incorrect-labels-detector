import numpy as np
import compare_module.compare as cmp


def test_cmp_unnes():
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
    assert boxes == exp_dicts


def test_cmp_lack():
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
    assert boxes == exp_dicts


def test_cmp_label():
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
    assert boxes == exp_dicts


def test_cmp_true():
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
    assert boxes == exp_dicts


def test_cmp_more_test_bbox():
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
    assert boxes == exp_dicts


def test_cmp_size():
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
    assert boxes == exp_dicts


def test_cmp_less_test_bbox():
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
    assert boxes == exp_dicts
