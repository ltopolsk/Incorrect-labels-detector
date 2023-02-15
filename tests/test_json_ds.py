from data.json_dataset import JsonDataSet
import numpy as np

ds = JsonDataSet('./outputs_dir')


def _test_json_ds(res, exp_vals):
    assert np.array_equal(res['boxes_mean'], exp_vals['boxes_mean'])
    assert np.array_equal(res['labels_mean'], exp_vals['labels_mean'])
    assert np.array_equal(res['boxes_test'], exp_vals['boxes_test'])
    assert np.array_equal(res['labels_test'], exp_vals['labels_test'])
    assert np.array_equal(res['errs'], exp_vals['errs'])


def _exp_vals(boxes_mean, labels_mean, boxes_test, labels_test, errs):
    return {'boxes_mean': boxes_mean,
            'labels_mean': labels_mean,
            'boxes_test': boxes_test,
            'labels_test': labels_test,
            'errs': errs}


def test_json_ds():
    exp_vals = _exp_vals(np.array([[53.06, 9.41, 277.63, 292.82]], dtype=np.float32),
                         np.array([11]),
                         np.array([[62.0, 55.0, 289.0, 283.0]]),
                         np.array([11]),
                         np.array([0]))
    _test_json_ds(ds[7], exp_vals)
    exp_vals = _exp_vals(np.array([[0.0, 77.73, 500.0, 317.64],
                                   [227.02, 56.54, 405.38, 222.63]], dtype=np.float32),
                         np.array([14, 11]),
                         np.array([[11.0, 7.0, 497.0, 351.0], [239.0, 47.0, 370.0, 194.0]]),
                         np.array([19, 8]),
                         np.array([-1, -1]))
    _test_json_ds(ds[0], exp_vals)
    exp_vals = _exp_vals(np.array([[56.57, 180.12, 283.38, 465.90]], dtype=np.float32),
                         np.array([18]),
                         np.array([[76.0, 40.0, 254.0, 429.0]]),
                         np.array([18]),
                         np.array([-2]))
    _test_json_ds(ds[8], exp_vals)
    exp_vals = _exp_vals(np.array([[234.15, 73.57, 372.43, 336.21],
                                   [50.36, 194.28, 239.23, 349.53],
                                   [22.23, 80.67, 211.59, 406.84]], dtype=np.float32),
                         np.array([8, 8, 17]),
                         np.array([[15.0, 191.0, 248.0, 363.0],
                                   [0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0]]),
                         np.array([8, -1, -1]),
                         np.array([-2, -4, -4]))
    _test_json_ds(ds[1], exp_vals)
    exp_vals = _exp_vals(np.array([[0, 0, 0, 0]], dtype=np.float32),
                         np.array([-1]),
                         np.array([[29, 30, 278, 357]]),
                         np.array([11]),
                         np.array([-5]))
    _test_json_ds(ds[5], exp_vals)
