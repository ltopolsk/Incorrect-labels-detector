from compare_module.iou import compute_IoU
# from compare_module.config import IOU_TRESHOLD, FUNC
from utils.config import opt
from data.json_dataset import JsonDataSet
from data.refer_voc_dataset import ReferVOCDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import torch as t
from tqdm import tqdm

OUTPUT_DIR = './outputs/5/mean/0_5/res/'
VOC_DIR = './VOCdevkit/VOC2007/'


def compute_img_detections(targets_ref, targets_json, stats):

    used_targs_ref = []

    def check_empty(stat_empty):
        if not len(targets_ref['boxes']):
            stats[stat_empty] += 1
            return 1

    def compare_test_refer(stats_incr, stat_empty):
        if check_empty(stat_empty): return
        _idx = np.where((targets_ref['boxes'] == targets_json['boxes_test'][i]).all(axis=1))[0]
        if len(_idx) != 0:
            idx = _idx[0]
            used_targs_ref.append(idx)
            stats[stats_incr[0]] += int(targets_ref['labels'][idx] == targets_json['labels_test'][i])
            stats[stats_incr[1]] += int(targets_ref['labels'][idx] != targets_json['labels_test'][i])
        else:
            stats[stat_empty] += 1

    def compare_mean_refer(stats_incr, stat_empty, ref_keyword):
        if check_empty(stat_empty): return
        overlaps = compute_IoU(t.unsqueeze(t.from_numpy(targets_json['boxes_mean'][i]), 0), t.from_numpy(targets_ref['boxes']))
        assigned_anno_idx = t.argmax(overlaps).numpy()
        if assigned_anno_idx.tolist() in used_targs_ref:
            stats[stats_incr[1]] += 1
        else:
            max_overlap = overlaps[assigned_anno_idx]
            if max_overlap >= opt.test_iou and targets_ref[ref_keyword][assigned_anno_idx]:
                stats[stats_incr[0]] += 1
                used_targs_ref.append(assigned_anno_idx.tolist())
            else:
                stats[stats_incr[1]] += 1

    def compare_test_not_ref(stats_incr, stat_empty):
        if check_empty(stat_empty): return
        _idx = np.where((targets_ref['boxes'] == targets_json['boxes_test'][i]).all(axis=1))[0]
        if len(_idx) == 0:
            stats[stats_incr[0]] += 1
        else:
            stats[stats_incr[1]] += 1
            used_targs_ref.append(_idx[0])

    def get_command_dict():
        return {0: (compare_test_refer, (('tp', 'fp'), 'fp')),
                -1: (compare_test_refer, (('fn', 'tn'), 'fn')),
                -2: (compare_mean_refer, (('tn', 'fn'), 'fn', 'resized')),
                -4: (compare_mean_refer, (('tn', 'fn'), 'tn', 'removed')),
                -5: (compare_test_not_ref, (('tn', 'fn'), 'tn')),
                }

    for i in range(len(targets_json['errs'])):
        command_dict = get_command_dict()
        func, args = command_dict[targets_json['errs'][i]]
        func(*args)

    if len(used_targs_ref) < np.array(targets_ref['boxes']).shape[0]:
        rest_idx = set(i for i in range(targets_ref['boxes'].shape[0])) - set(used_targs_ref)
        stats['fp'] += len(rest_idx)

def compute():
    json_ds = JsonDataSet(OUTPUT_DIR)
    refer_ds = ReferVOCDataset(VOC_DIR, split='test')
    stats = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for i in tqdm(range(len(json_ds)), desc='Computing detections'):
        compute_img_detections(refer_ds[i], json_ds[i], stats)
    metrics = {'acc': ((stats["tp"] + stats["tn"])/(stats["tp"]+stats["tn"]+stats["fp"]+stats["fn"])),
               'prec': (stats["tp"]/(stats["tp"]+stats["fp"])),
               'prec_negative': (stats["tn"]/(stats["tn"]+stats["fn"])),
               'recall': (stats["tp"]/(stats["tp"]+stats["fn"])),
               'recall_negative': (stats["tn"]/(stats["tn"]+stats["fp"])),
               'fpr':(stats["fp"]/(stats["tn"]+stats["fp"])),
               'fnr': (stats["fn"]/(stats["tp"]+stats["fn"])),}
    file_name = '0_' + str(int(opt.test_iou*100)) + '_' + ('mean_box' if opt.use_mean else 'nms')
    # if not os.path.exists(opt.output_dir):
        # os.makedirs(opt.output_dir)
    with open(os.path.join('.\\confusion-mat', '5', 'metrics', f'{file_name}.txt'), 'w') as f:
        for k, v in metrics.items():
            f.write(f'{k}: {v:.3f}\n')
    cm = np.array([[stats['tp'], stats['fp']], [stats['fn'], stats['tn']]])
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    cm_labels = {
        (0, 0): 'tp',
        (0, 1): 'fp',
        (1, 0): 'fn',
        (1, 1): 'tn'
    }
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm_labels[(i, j)]} = {cm[i, j]}', color='w')
    plt.savefig(os.path.join('.\\confusion-mat', '5', 'imgs', f'{file_name}.png'))
    plt.close()

if __name__ == "__main__":
    compute()
