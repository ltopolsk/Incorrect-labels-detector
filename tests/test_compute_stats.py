from compute_stats import compute_img_detections
from data.json_dataset import JsonDataSet
from data.refer_voc_dataset import ReferVOCDataset


OUTPUT_DIR = './outputs_dir'
VOC_DIR = './data_dir'

json_ds = JsonDataSet(OUTPUT_DIR)
refer_ds = ReferVOCDataset(VOC_DIR, split='test')


def _test_err(refer, json, exp_stats):
    stats = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, }
    compute_img_detections(refer, json, stats)
    assert stats == exp_stats


def test_err_0_tp():
    _test_err(refer_ds[3], json_ds[3], {'tp': 1, 'tn': 0, 'fp': 0, 'fn': 0, })


def test_err_0_fp():
    _test_err(refer_ds[3], json_ds[5], {'tp': 0, 'tn': 0, 'fp': 2, 'fn': 0, })


def test_err_1_tn():
    _test_err(refer_ds[0], json_ds[0], {'tp': 0, 'tn': 2, 'fp': 0, 'fn': 0, })


def test_err_1_fn():
    _test_err(refer_ds[0], json_ds[6], {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 2, })


def test_err_2_tn():
    _test_err(refer_ds[6], json_ds[7], {'tp': 0, 'tn': 1, 'fp': 0, 'fn': 0, })


def test_err_2_fn():
    _test_err(refer_ds[4], json_ds[4], {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 1, })