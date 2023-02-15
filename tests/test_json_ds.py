from data.json_dataset import JsonDataSet
import numpy as np

ds = JsonDataSet('./outputs_dir')


def test_json_true():

    res = ds[7]
    assert np.array_equal(res['boxes_mean'], np.array([[53.06, 9.41, 277.63, 292.82]], dtype=np.float32))
    assert np.array_equal(res['labels_mean'], np.array([11]))
    assert np.array_equal(res['boxes_test'], np.array([[62.0, 55.0, 289.0, 283.0]]))
    assert np.array_equal(res['labels_test'], np.array([11]))
    assert np.array_equal(res['errs'], np.array([0]))


def test_json_err_label():

    res = ds[0]
    assert np.array_equal(res['boxes_mean'], np.array([[0.0, 77.73, 500.0, 317.64], [227.02, 56.54, 405.38, 222.63]], dtype=np.float32))
    assert np.array_equal(res['labels_mean'], np.array([14, 11]))
    assert np.array_equal(res['boxes_test'], np.array([[11.0, 7.0, 497.0, 351.0], [239.0, 47.0, 370.0, 194.0]]))
    assert np.array_equal(res['labels_test'], np.array([19, 8]))
    assert np.array_equal(res['errs'], np.array([-1, -1]))


def test_json_err_bbox_size():

    res = ds[8]
    assert np.array_equal(res['boxes_mean'], np.array([[56.57, 180.12, 283.38, 465.90]], dtype=np.float32))
    assert np.array_equal(res['labels_mean'], np.array([18]))
    assert np.array_equal(res['boxes_test'], np.array([[76.0, 40.0, 254.0, 429.0]]))
    assert np.array_equal(res['labels_test'], np.array([18]))
    assert np.array_equal(res['errs'], np.array([-2]))


def test_json_err_lack_bbox():

    res = ds[1]
    assert np.array_equal(res['boxes_mean'], np.array([[234.15, 73.57, 372.43, 336.21],
                                                       [50.36, 194.28, 239.23, 349.53],
                                                       [22.23, 80.67, 211.59, 406.84]], dtype=np.float32))
    assert np.array_equal(res['labels_mean'], np.array([8, 8, 17]))
    assert np.array_equal(res['boxes_test'], np.array([[15.0, 191.0, 248.0, 363.0],
                                                       [0.0, 0.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0, 0.0]]))
    assert np.array_equal(res['labels_test'], np.array([8, -1, -1]))
    assert np.array_equal(res['errs'], np.array([-2, -4, -4]))


def test_json_err_bbox_unnes():

    res = ds[5]
    assert np.array_equal(res['boxes_mean'], np.array([[0, 0, 0, 0]], dtype=np.float32))
    assert np.array_equal(res['labels_mean'], np.array([-1]))
    assert np.array_equal(res['boxes_test'], np.array([[29.0, 30.0, 278.0, 357.0]]))
    assert np.array_equal(res['labels_test'], np.array([11]))
    assert np.array_equal(res['errs'], np.array([-5]))