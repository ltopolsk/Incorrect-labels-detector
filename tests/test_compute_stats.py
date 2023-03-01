from compute_stats import compute_img_detections
from data.json_dataset import JsonDataSet
from data.refer_voc_dataset import ReferVOCDataset


OUTPUT_DIR = './outputs_dir'
VOC_DIR = './data_dir'

def test_comp_img_stats():
    json_ds = JsonDataSet(OUTPUT_DIR)
    refer_ds = ReferVOCDataset(VOC_DIR, split='test')
    