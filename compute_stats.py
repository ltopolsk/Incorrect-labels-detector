from compare_module.iou import compute_IoU
from compare_module.config import IOU_TRESHOLD, FUNC
from data.json_dataset import JsonDataSet
from data.refer_voc_dataset import ReferVOCDataset
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = './outputs/res/'
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
        overlaps = compute_IoU(np.expand_dims(targets_json['boxes_mean'][i], axis=0), targets_ref['boxes'])
        assigned_anno_idx = np.argmax(overlaps)
        if assigned_anno_idx in used_targs_ref:
            stats[stats_incr[1]] += 1
        else:
            max_overlap = overlaps[assigned_anno_idx]
            if max_overlap >= IOU_TRESHOLD and targets_ref[ref_keyword][assigned_anno_idx]:
                stats[stats_incr[0]] += 1
                used_targs_ref.append(assigned_anno_idx)
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


if __name__ == "__main__":
    json_ds = JsonDataSet(OUTPUT_DIR)
    refer_ds = ReferVOCDataset(VOC_DIR, split='test')
    stats = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for i in range(len(json_ds)):
        compute_img_detections(refer_ds[i], json_ds[i], stats)
    metrics = {'acc': ((stats["tp"] + stats["tn"])/(stats["tp"]+stats["tn"]+stats["fp"]+stats["fn"])),
               'prec': (stats["tp"]/(stats["tp"]+stats["fp"])),
               'prec_negative': (stats["tn"]/(stats["tn"]+stats["fn"])),
               'recall': (stats["tp"]/(stats["tp"]+stats["fn"])),
               'recall_negative': (stats["tn"]/(stats["tn"]+stats["fp"])), }
    file_name = '0_' + str(int(IOU_TRESHOLD*100)) + '_' + FUNC.__name__ if FUNC is not None else 'nms'
    with open(os.path.join('./confusion-mat', 'metrics', f'{file_name}.txt'), 'w') as f:
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
    plt.savefig(os.path.join('./confusion-mat', 'imgs', f'{file_name}.png'))
