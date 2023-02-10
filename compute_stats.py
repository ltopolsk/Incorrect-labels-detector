from compare_module.compare import IOU_TRESHOLD, compute_IoU
from data.json_dataset import JsonDataSet
from data.refer_voc_dataset import ReferVOCDataset
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = './outputs/res/'
VOC_DIR = './VOCdevkit/VOC2007/'

tp = 0
tn = 0
fp = 0
fn = 0


def get_targets(ds, i):
    _, _bboxes, _labels, _ = ds[i]
    return {'boxes': _bboxes, 'labels': _labels}


def compute_img_detections(targets_ref, targets_json):
    global tp
    global tn
    global fp
    global fn
    used_targs_ref = []
    for i in range(len(targets_json['errs'])):
        if targets_json['errs'][i] == 0:
            _idx = np.where((targets_ref['boxes'] == targets_json['boxes_test'][i]).all(axis=1))[0]
            if len(_idx) != 0:
                used_targs_ref.append(_idx)
                idx = _idx[0]
                if targets_ref['labels'][idx] == targets_json['labels_test'][i]:
                    tp += 1
                else:
                    fp += 1
            else:
                fp += 1
        elif targets_json['errs'][i] == -1:
            _idx = np.where((targets_ref['boxes'] == targets_json['boxes_test'][i]).all(axis=1))[0]
            if len(_idx) != 0:
                used_targs_ref.append(_idx)
                idx = _idx[0]
                if targets_ref['labels'][idx] != targets_json['labels_test'][i]:
                    tn += 1
                else:
                    fn += 1
            else:
                fn += 1
        elif targets_json['errs'][i] == -2:
            overlaps = compute_IoU(np.expand_dims(targets_json['boxes_mean'][i], axis=0), targets_ref['boxes'])
            assigned_anno_idx = np.argmax(overlaps, axis=1)
            if assigned_anno_idx in used_targs_ref:
                fn += 1
            else:
                max_overlap = overlaps[0, assigned_anno_idx]
                if max_overlap >= IOU_TRESHOLD and targets_ref['resized'][assigned_anno_idx] == 1:
                    tn += 1
                    used_targs_ref.append(assigned_anno_idx)
                else:
                    fn += 1

        elif targets_json['errs'][i] == -3:
            pass

        elif targets_json['errs'][i] == -4:
            overlaps = compute_IoU(np.expand_dims(targets_json['boxes_mean'][i], axis=0), targets_ref['boxes'])
            assigned_anno_idx = np.argmax(overlaps, axis=1)
            if assigned_anno_idx in used_targs_ref:
                fn += 1
            else:
                max_overlap = overlaps[0, assigned_anno_idx]
                if max_overlap >= IOU_TRESHOLD and targets_ref['removed'][assigned_anno_idx] == 1:
                    used_targs_ref.append(assigned_anno_idx)
                    tn += 1
                else:
                    fn += 1

        elif targets_json['errs'][i] == -5:
            _idx = np.where((targets_ref['boxes'] == targets_json['boxes_test'][i]).all(axis=1))[0]
            if len(_idx) == 0:
                tn += 1
            else:
                fn += 1
                used_targs_ref.append(_idx)
    if targets_json['boxes_mean'].shape[0] < targets_ref['boxes'].shape[0]:
        rest_idx = set([i for i in range(targets_ref['boxes'].shape[0])])
        used_idx_set = set()
        for idx in used_targs_ref:
            used_idx_set.add(idx[0])
        rest_idx = rest_idx - used_idx_set
        fp += len(rest_idx)


def draw_cm(cm):
    pass


if __name__ == "__main__":
    json_ds = JsonDataSet(OUTPUT_DIR)
    refer_ds = ReferVOCDataset(VOC_DIR, split='test')
    for i in range(len(json_ds)):
        compute_img_detections(refer_ds[i], json_ds[i])
    print(f'acc: {(tp+tn)/(tp+tn+fp+fn):.4f}')
    print(f'prec: {tp/(tp+fp):.4f}')
    print(f'prec_negative: {tn/(tn+fn):.4f}')
    print(f'recall: {tp/(tp+fn):.4f}')
    print(f'recall_negative: {tn/(tn+fp):.4f}')
    cm = np.array([[tp, fp], [fn, tn]])
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

