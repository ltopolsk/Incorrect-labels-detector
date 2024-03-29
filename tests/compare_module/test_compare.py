import torch as t
from compare_module.compare import compare
import compare_module.utils as c


def test_cmp_unnes():
    boxes = compare(t.tensor([]),
                    t.tensor([]),
                    t.tensor([[15., 20., 60., 100],
                              [50., 10., 100., 30]]),
                    t.tensor([1, 10]))

    exp_dicts = [{'box_mean': None,
                  'label_mean': None,
                  'box_test': [15., 20., 60., 100],
                  'label_test': 1,
                  'err': c.ERR_BBOX_UNNES, },
                 {'box_mean': None,
                  'label_mean': None,
                  'box_test': [50., 10., 100., 30],
                  'label_test': 10,
                  'err': c.ERR_BBOX_UNNES, }]
    assert boxes == exp_dicts


def test_cmp_lack():
    boxes = compare(t.tensor([[15., 20., 60., 100],
                              [50., 10., 100., 30]]),
                    t.tensor([1, 10]),
                    t.tensor([]),
                    t.tensor([]))

    exp_dicts = [{'box_mean': [15., 20., 60., 100],
                  'label_mean': 1,
                  'box_test': None,
                  'label_test': None,
                  'err': c.ERR_LACK_BBOX, },
                 {'box_mean': [50., 10., 100., 30],
                  'label_mean': 10,
                  'box_test': None,
                  'label_test': None,
                  'err': c.ERR_LACK_BBOX, }]
    assert boxes == exp_dicts


def test_cmp_label():
    boxes = compare(t.tensor([[15., 20., 60., 100],
                              [50., 10., 100., 30]]),
                    t.tensor([1, 10]),
                    t.tensor([[15., 20., 60., 100],
                              [50., 10., 100., 30]]),
                    t.tensor([10, 9]))

    exp_dicts = [{'box_mean': [15., 20., 60., 100],
                  'label_mean': 1,
                  'box_test': [15., 20., 60., 100],
                  'label_test': 10,
                  'err': c.ERR_LABEL, },
                 {'box_mean': [50., 10., 100., 30],
                  'label_mean': 10,
                  'box_test': [50., 10., 100., 30],
                  'label_test': 9,
                  'err': c.ERR_LABEL, }]
    assert boxes == exp_dicts


def test_cmp_true():
    boxes = compare(t.tensor([[15., 20., 60., 100],
                              [50., 10., 100., 30]]),
                    t.tensor([1, 10]),
                    t.tensor([[15., 20., 60., 100],
                              [50., 10., 100., 30]]),
                    t.tensor([1, 10]))

    exp_dicts = [{'box_mean': [15., 20., 60., 100],
                  'label_mean': 1,
                  'box_test': [15., 20., 60., 100],
                  'label_test': 1,
                  'err': c.TRUE_POS, },
                 {'box_mean': [50., 10., 100., 30],
                  'label_mean': 10,
                  'box_test': [50., 10., 100., 30],
                  'label_test': 10,
                  'err': c.TRUE_POS, }]
    assert boxes == exp_dicts


def test_cmp_more_test_bbox():
    boxes = compare(t.tensor([[50., 10., 100., 30]]),
                    t.tensor([10]),
                    t.tensor([[15., 20., 60., 100],
                              [50., 10., 100., 30]]),
                    t.tensor([1, 10]))

    exp_dicts = [{'box_mean': [50., 10., 100., 30],
                  'label_mean': 10,
                  'box_test': [50., 10., 100., 30],
                  'label_test': 10,
                  'err': c.TRUE_POS, },
                 {'box_mean': None,
                  'label_mean': None,
                  'box_test': [15., 20., 60., 100],
                  'label_test': 1,
                  'err': c.ERR_BBOX_UNNES, }, ]
    assert boxes == exp_dicts


def test_cmp_size():
    boxes = compare(t.tensor([[15., 10., 300., 150],
                              [50., 10., 55., 15]]),
                    t.tensor([1, 10]),
                    t.tensor([[15., 20., 60., 100],
                              [50., 10., 100., 30]]),
                    t.tensor([1, 10]))

    exp_dicts = [{'box_mean': [15., 10., 300., 150],
                  'label_mean': 1,
                  'box_test': [15., 20., 60., 100],
                  'label_test': 1,
                  'err': c.ERR_BBOX_SIZE, },
                 {'box_mean': [50., 10., 55., 15],
                  'label_mean': 10,
                  'box_test': [50., 10., 100., 30],
                  'label_test': 10,
                  'err': c.ERR_BBOX_SIZE, }]
    assert boxes == exp_dicts


def test_cmp_less_test_bbox():
    boxes = compare(t.tensor([[15., 20., 60., 100],
                              [15., 20., 60., 100]]),
                    t.tensor([1, 10]),
                    t.tensor([[15., 20., 60., 100]]),
                    t.tensor([1]))

    exp_dicts = [{'box_mean': [15., 20., 60., 100],
                  'label_mean': 1,
                  'box_test': [15., 20., 60., 100],
                  'label_test': 1,
                  'err': c.TRUE_POS, },
                 {'box_mean':  [15., 20., 60., 100],
                  'label_mean': 10,
                  'box_test': None,
                  'label_test': None,
                  'err': c.ERR_LACK_BBOX, }]
    assert boxes == exp_dicts
