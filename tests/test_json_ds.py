from data.json_dataset import JsonDataSet
import numpy as np

def test_json_ds():
    ds = JsonDataSet('./outputs_dir')
    res = ds[7]
    assert np.array_equal(res['boxes_mean'], np.array([[53.06, 9.41, 277.63, 292.82]], dtype=np.float32))
    assert np.array_equal(res['labels_mean'], np.array([11]))
    assert np.array_equal(res['boxes_test'], np.array([[62.0, 55.0, 289.0, 283.0]]))
    assert np.array_equal(res['labels_test'], np.array([11]))
    assert np.array_equal(res['errs'], np.array([0]))
