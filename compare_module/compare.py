import torch as t
import compare_module.utils as c
from .iou import compute_IoU


def compare(bboxes_mean, labels_mean, bboxes_test, labels_test, scores_mean, iou=0.5):
    ret = []
    if bboxes_mean.shape[0] == 0 and bboxes_test.shape[0] > 0:
        for bbox, label in zip(bboxes_test, labels_test):
            ret.append({
                'box_mean': None,
                'label_mean': None,
                'box_test': bbox.tolist(),
                'label_test': label.item(),
                'err': c.ERR_BBOX_UNNES,
                'score':0,
            })
        return ret

    detected_anno = []
    for i, item in enumerate(bboxes_mean):
        if bboxes_test.shape[0] == 0:
            ret.append({
                'box_mean': item.tolist(),
                'label_mean': labels_mean[i].item(),
                'box_test': None,
                'label_test': -1,
                'err': c.ERR_LACK_BBOX,
                'score':scores_mean[i].item(),
            })
            continue
        overlaps = compute_IoU(item.unsqueeze(0), bboxes_test)
        assigned_anno_idx = t.argmax(overlaps)
        max_overlap = overlaps[assigned_anno_idx]
        assigned_label = labels_test[assigned_anno_idx]
        assigned_anno = bboxes_test[assigned_anno_idx]
        if assigned_anno_idx in detected_anno:
            ret.append({
                'box_mean': item.tolist(),
                'label_mean': labels_mean[i].item(),
                'box_test': None,
                'label_test': -1,
                'err': c.ERR_LACK_BBOX,
                'score':scores_mean[i].item(),
            })
            continue
        if max_overlap < iou:
            ret.append({
                'box_mean': item.tolist(),
                'label_mean': labels_mean[i].item(),
                'box_test': assigned_anno.tolist(),
                'label_test': assigned_label.item(),
                'err': c.ERR_BBOX_SIZE,
                'score':scores_mean[i].item(),
            })
            detected_anno.append(assigned_anno_idx)
            continue
        if labels_mean[i] != assigned_label:
            ret.append({
                'box_mean': item.tolist(),
                'label_mean': labels_mean[i].item(),
                'box_test': assigned_anno.tolist(),
                'label_test': assigned_label.item(),
                'err': c.ERR_LABEL,
                'score':scores_mean[i].item(),
            })
            detected_anno.append(assigned_anno_idx)
            continue
        ret.append({
                'box_mean': item.tolist(),
                'label_mean': labels_mean[i].item(),
                'box_test': assigned_anno.tolist(),
                'label_test': assigned_label.item(),
                'err': c.TRUE_POS,
                'score':scores_mean[i].item(),
        })
        detected_anno.append(assigned_anno_idx)
    if len(detected_anno) < bboxes_test.shape[0]:
        rest_idx = [t.tensor(i) for i in range(len(bboxes_test)) if t.tensor(i) not in detected_anno]
        rest_bboxes = [bboxes_test[idx] for idx in rest_idx]
        rest_labels = [labels_test[idx] for idx in rest_idx]
        for bbox, label in zip(rest_bboxes, rest_labels):
            ret.append({
                'box_mean': None,
                'label_mean': None,
                'box_test': bbox.tolist(),
                'label_test': label.item(),
                'err': c.ERR_BBOX_UNNES,
                'score':0,
            })

    return sorted(ret, key=lambda x: x["label_test"], reverse=True)
