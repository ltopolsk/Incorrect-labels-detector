import numpy as np
import compare_module.config as c
from .iou import compute_IoU


def compare(bboxes_mean, labels_mean, bboxes_test, labels_test):
    t = []
    if bboxes_mean.shape[0] == 0 and bboxes_test.shape[0] > 0:
        for bbox, label in zip(bboxes_test, labels_test):
            t.append({
                'box_mean': None,
                'label_mean': None,
                'box_test': bbox.tolist(),
                'label_test': label.item(),
                'err': c.ERR_BBOX_UNNES,
            })
        return t

    detected_anno = []
    for i, item in enumerate(bboxes_mean):
        if bboxes_test.shape[0] == 0:
            t.append({
                'box_mean': item.tolist(),
                'label_mean': labels_mean[i].item(),
                'box_test': None,
                'label_test': None,
                'err': c.ERR_LACK_BBOX,
            })
            continue
        overlaps = compute_IoU(np.expand_dims(item, axis=0), bboxes_test)
        assigned_anno_idx = np.argmax(overlaps)
        max_overlap = overlaps[assigned_anno_idx]
        assigned_label = labels_test[assigned_anno_idx]
        assigned_anno = bboxes_test[assigned_anno_idx]
        if assigned_anno_idx in detected_anno:
            t.append({
                'box_mean': item.tolist(),
                'label_mean': labels_mean[i].item(),
                'box_test': None,
                'label_test': None,
                'err': c.ERR_LACK_BBOX,
            })
            continue
        if max_overlap < c.IOU_TRESHOLD:
            t.append({
                'box_mean': item.tolist(),
                'label_mean': labels_mean[i].item(),
                'box_test': assigned_anno.tolist(),
                'label_test': assigned_label.item(),
                'err': c.ERR_BBOX_SIZE,
            })
            detected_anno.append(assigned_anno_idx)
            continue
        if labels_mean[i] != assigned_label:
            t.append({
                'box_mean': item.tolist(),
                'label_mean': labels_mean[i].item(),
                'box_test': assigned_anno.tolist(),
                'label_test': assigned_label.item(),
                'err': c.ERR_LABEL,
            })
            detected_anno.append(assigned_anno_idx)
            continue
        t.append({
                'box_mean': item.tolist(),
                'label_mean': labels_mean[i].item(),
                'box_test': assigned_anno.tolist(),
                'label_test': assigned_label.item(),
                'err': c.TRUE_POS,
        })
        detected_anno.append(assigned_anno_idx)
    if len(detected_anno) < bboxes_test.shape[0]:
        rest_idx = set([i for i in range(len(bboxes_test))]) - set(detected_anno)
        rest_bboxes = [bboxes_test[idx] for idx in rest_idx]
        rest_labels = [labels_test[idx] for idx in rest_idx]
        for bbox, label in zip(rest_bboxes, rest_labels):
            t.append({
                'box_mean': None,
                'label_mean': None,
                'box_test': bbox.tolist(),
                'label_test': label.item(),
                'err': c.ERR_BBOX_UNNES,
            })

    return t

