from data.json_dataset import JsonDataSet
import numpy as np

ds = JsonDataSet('./outputs_dir')


exp_vals_0 = {'boxes_mean': np.array([[53.06, 9.41, 277.63, 292.82]],
                                     dtype=np.float32),
              'labels_mean': np.array([11]),
              'boxes_test': np.array([[62.0, 55.0, 289.0, 283.0]],
                                     dtype=np.float32),
              'labels_test': np.array([11]),
              'errs': np.array([0])}

exp_vals_1 = {'boxes_mean': np.array([[0.0, 77.73, 500.0, 317.64],
                                      [227.02, 56.54, 405.38, 222.63]],
                                     dtype=np.float32),
              'labels_mean': np.array([14, 11]),
              'boxes_test': np.array([[11.0, 7.0, 497.0, 351.0],
                                      [239.0, 47.0, 370.0, 194.0]],
                                     dtype=np.float32),
              'labels_test': np.array([19, 8]),
              'errs': np.array([-1, -1])}

exp_vals_2 = {'boxes_mean': np.array([[56.57, 180.12, 283.38, 465.90],
                                      [52.57, 175.12, 280.38, 460.90]],
                                     dtype=np.float32),
              'labels_mean': np.array([18, 18]),
              'boxes_test': np.array([[76.0, 40.0, 254.0, 429.0],
                                      [76.0, 40.0, 254.0, 429.0]],
                                     dtype=np.float32),
              'labels_test': np.array([18, 18]),
              'errs': np.array([-2, -2])}

exp_vals_3 = {'boxes_mean': np.array([[234.15, 73.57, 372.43, 336.21],
                                      [50.36, 194.28, 239.23, 349.53],
                                      [22.23, 80.67, 211.59, 406.84]],
                                     dtype=np.float32),
              'labels_mean': np.array([8, 8, 17]),
              'boxes_test': np.array([[15.0, 191.0, 248.0, 363.0],
                                      [0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0]],
                                     dtype=np.float32),
              'labels_test': np.array([8, -1, -1]),
              'errs': np.array([-2, -4, -4])}

exp_vals_4 = {'boxes_mean': np.array([[0, 0, 0, 0]],
                                     dtype=np.float32),
              'labels_mean': np.array([-1]),
              'boxes_test': np.array([[29, 30, 278, 357]],
                                     dtype=np.float32),
              'labels_test': np.array([11]),
              'errs': np.array([-5])}


def _test_json_ds(res, exp_vals):
    assert np.array_equal(res['boxes_mean'], exp_vals['boxes_mean'])
    assert np.array_equal(res['labels_mean'], exp_vals['labels_mean'])
    assert np.array_equal(res['boxes_test'], exp_vals['boxes_test'])
    assert np.array_equal(res['labels_test'], exp_vals['labels_test'])
    assert np.array_equal(res['errs'], exp_vals['errs'])


def test_err_0():
    _test_json_ds(ds[3], exp_vals_0)


def test_err_1():
    _test_json_ds(ds[0], exp_vals_1)


def test_err_2():
    _test_json_ds(ds[4], exp_vals_2)


def test_err_3():
    _test_json_ds(ds[1], exp_vals_3)


def test_err_4():
    _test_json_ds(ds[2], exp_vals_4)
