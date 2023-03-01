from compare_module.compare import IOU_TRESHOLD, compute_IoU
from data.json_dataset import JsonDataSet
from data.refer_voc_dataset import ReferVOCDataset
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = './outputs/res/'
VOC_DIR = './VOCdevkit/VOC2007/'


def compute_img_detections(targets_ref, targets_json, stats):
    # global stats
    used_targs_ref = []
    for i in range(len(targets_json['errs'])):
        if targets_json['errs'][i] == 0:
            _idx = np.where((targets_ref['boxes'] == targets_json['boxes_test'][i]).all(axis=1))[0]
            if len(_idx) != 0:
                idx = _idx[0]
                used_targs_ref.append(idx)
                if targets_ref['labels'][idx] == targets_json['labels_test'][i]:
                    stats['tp'] += 1
                else:
                    stats['fp'] += 1
            else:
                stats['fp'] += 1
        elif targets_json['errs'][i] == -1:
            _idx = np.where((targets_ref['boxes'] == targets_json['boxes_test'][i]).all(axis=1))[0]
            if len(_idx) != 0:
                idx = _idx[0]
                used_targs_ref.append(idx)
                if targets_ref['labels'][idx] != targets_json['labels_test'][i]:
                    stats['tn'] += 1
                else:
                    stats['fn'] += 1
            else:
                stats['fn'] += 1
        elif targets_json['errs'][i] == -2:
            overlaps = compute_IoU(np.expand_dims(targets_json['boxes_mean'][i], axis=0), targets_ref['boxes'])
            assigned_anno_idx = np.argmax(overlaps)
            if assigned_anno_idx in used_targs_ref:
                stats['fn'] += 1
            else:
                max_overlap = overlaps[assigned_anno_idx]
                if max_overlap >= IOU_TRESHOLD and targets_ref['resized'][assigned_anno_idx] == 1:
                    stats['tn'] += 1
                    used_targs_ref.append(assigned_anno_idx)
                else:
                    stats['fn'] += 1

        elif targets_json['errs'][i] == -3:
            pass

        elif targets_json['errs'][i] == -4:
            overlaps = compute_IoU(np.expand_dims(targets_json['boxes_mean'][i], axis=0), targets_ref['boxes'])
            assigned_anno_idx = np.argmax(overlaps)
            if assigned_anno_idx in used_targs_ref:
                stats['fn'] += 1
            else:
                max_overlap = overlaps[assigned_anno_idx]
                if max_overlap >= IOU_TRESHOLD and targets_ref['removed'][assigned_anno_idx] == 1:
                    used_targs_ref.append(assigned_anno_idx)
                    stats['tn'] += 1
                else:
                    stats['fn'] += 1

        elif targets_json['errs'][i] == -5:
            _idx = np.where((targets_ref['boxes'] == targets_json['boxes_test'][i]).all(axis=1))[0]
            if len(_idx) == 0:
                stats['tn'] += 1
            else:
                stats['fn'] += 1
                used_targs_ref.append(_idx[0])
    if targets_json['boxes_mean'].shape[0] < targets_ref['boxes'].shape[0]:
        rest_idx = set([i for i in range(targets_ref['boxes'].shape[0])])
        used_idx_set = set()
        for idx in used_targs_ref:
            used_idx_set.add(idx)
        rest_idx = rest_idx - used_idx_set
        stats['fp'] += len(rest_idx)


if __name__ == "__main__":
    json_ds = JsonDataSet(OUTPUT_DIR)
    refer_ds = ReferVOCDataset(VOC_DIR, split='test')
    stats = {'tp': 0,'tn': 0,'fp': 0,'fn': 0,}
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
