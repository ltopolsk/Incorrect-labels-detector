import numpy as np
from data.refer_voc_dataset import ReferVOCDataset


def test_refer_ds():
    ds = ReferVOCDataset(data_dir='./data_dir/', split='test')
    targets = ds[0]
    assert np.array_equal(targets['boxes'], np.array([[135., 76., 357., 359.]]))
    assert np.array_equal(targets['labels'], np.array([1]))
    assert np.array_equal(targets['difficult'], np.array([0]))
    assert np.array_equal(targets['renamed'], np.array([0]))
    assert np.array_equal(targets['resized'], np.array([0]))
    assert np.array_equal(targets['removed'], np.array([0]))
