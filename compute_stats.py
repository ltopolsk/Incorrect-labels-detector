from compare_module.compare import IOU_TRESHOLD, compute_IoU
from data.json_dataset import JsonDataSet
from data.refer_voc_dataset import ReferVOCDataset
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = './outputs/res/'
VOC_DIR = './VOCdevkit/VOC2007/'


def compute_img_detections(targets_ref, targets_json, stats):

    def compare_test_refer(pos, stats_incr, stat_empty):
        if not len(targets_ref['boxes']):
            stats[stat_empty] += 1
            return

        _idx = np.where((targets_ref['boxes'] == targets_json['boxes_test'][pos]).all(axis=1))[0]
        if len(_idx) != 0:
            idx = _idx[0]
            used_targs_ref.append(idx)
            stats[stats_incr[0]] += int(targets_ref['labels'][idx] == targets_json['labels_test'][pos])
            stats[stats_incr[1]] += int(targets_ref['labels'][idx] != targets_json['labels_test'][pos])
        else:
            stats[stat_empty] += 1

    def compare_mean_refer(pos, stats_incr, stat_empty, ref_keyword):
        if not len(targets_ref['boxes']):
            stats[stat_empty] += 1
            return
        overlaps = compute_IoU(np.expand_dims(targets_json['boxes_mean'][pos], axis=0), targets_ref['boxes'])
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

    def compare_test_not_ref(pos, stats_incr, empty_stat):
        if not len(targets_ref['boxes']):
            stats[empty_stat] += 1
            return
        _idx = np.where((targets_ref['boxes'] == targets_json['boxes_test'][pos]).all(axis=1))[0]
        if len(_idx) == 0:
            stats[stats_incr[0]] += 1
        else:
            stats[stats_incr[1]] += 1
            used_targs_ref.append(_idx[0])
    used_targs_ref = []
    for i in range(len(targets_json['errs'])):
        if targets_json['errs'][i] == 0:
            compare_test_refer(i, ('tp', 'fp'), 'fp')

        elif targets_json['errs'][i] == -1:
            compare_test_refer(i, ('fn', 'tn'), 'fn')

        elif targets_json['errs'][i] == -2:
            compare_mean_refer(i, ('tn', 'fn'), 'fn', 'resized')

        elif targets_json['errs'][i] == -4:
            compare_mean_refer(i, ('tn', 'fn'), 'tn', 'removed')

        elif targets_json['errs'][i] == -5:
            compare_test_not_ref(i, ('tn', 'fn'), 'tn')

    if len(used_targs_ref) < np.array(targets_ref['boxes']).shape[0]:
        rest_idx = set(i for i in range(targets_ref['boxes'].shape[0]))
        used_idx_set = set(used_targs_ref)
        rest_idx = rest_idx - used_idx_set
        stats['fp'] += len(rest_idx)


if __name__ == "__main__":
    json_ds = JsonDataSet(OUTPUT_DIR)
    refer_ds = ReferVOCDataset(VOC_DIR, split='test')
    stats = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for i in range(len(json_ds)):
        compute_img_detections(refer_ds[i], json_ds[i], stats)
    print(f'acc: {(stats["tp"] + stats["tn"])/(stats["tp"]+stats["tn"]+stats["fp"]+stats["fn"]):.4f}')
    print(f'prec: {stats["tp"]/(stats["tp"]+stats["fp"]):.4f}')
    print(f'prec_negative: {stats["tn"]/(stats["tn"]+stats["fn"]):.4f}')
    print(f'recall: {stats["tp"]/(stats["tp"]+stats["fn"]):.4f}')
    print(f'recall_negative: {stats["tn"]/(stats["tn"]+stats["fp"]):.4f}')
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
    plt.show()
