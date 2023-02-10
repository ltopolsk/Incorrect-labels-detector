import numpy as np


IOU_TRESHOLD = 0.5

ERR_BBOX_UNNES = -5
ERR_LACK_BBOX = -4
ERR_REDUN = -3
ERR_BBOX_SIZE = -2
ERR_LABEL = -1
TRUE_POS = 0


def compute_IoU(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compare(bboxes_mean, labels_mean, bboxes_test, labels_test):
    t = []
    if bboxes_mean.shape[0] == 0 and bboxes_test.shape[0] > 0:
        for bbox, label in zip(bboxes_test, labels_test):
            t.append({
                'box_mean': None,
                'label_mean': None,
                'box_test': bbox.tolist(),
                'label_test': label.item(),
                'err': ERR_BBOX_UNNES,
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
                'err': ERR_LACK_BBOX,
            })
            continue
        overlaps = compute_IoU(item, bboxes_test)
        assigned_anno_idx = np.argmax(overlaps, axis=1)
        max_overlap = overlaps[0, assigned_anno_idx]
        assigned_label = labels_test[assigned_anno_idx]
        assigned_anno = bboxes_test[assigned_anno_idx]
        if assigned_anno_idx in detected_anno:
            t.append({
                'box_mean': item.tolist(),
                'label_mean': labels_mean[i].item(),
                'box_test': None,
                'label_test': None,
                'err': ERR_LACK_BBOX,
            })
            continue
        if max_overlap < IOU_TRESHOLD:
            t.append({
                'box_mean': item.tolist(),
                'label_mean': labels_mean[i].item(),
                'box_test': assigned_anno.tolist(),
                'label_test': assigned_label.item(),
                'err': ERR_BBOX_SIZE,
            })
            detected_anno.append(assigned_anno_idx)
            continue
        if labels_mean[i] != assigned_label:
            t.append({
                'box_mean': item.tolist(),
                'label_mean': labels_mean[i].item(),
                'box_test': assigned_anno.tolist(),
                'label_test': assigned_label.item(),
                'err': ERR_LABEL,
            })
            detected_anno.append(assigned_anno_idx)
            continue
        t.append({
                'box_mean': item.tolist(),
                'label_mean': labels_mean[i].item(),
                'box_test': assigned_anno.tolist(),
                'label_test': assigned_label.item(),
                'err': TRUE_POS,
        })
        detected_anno.append(assigned_anno_idx)
    if bboxes_mean.shape[0] < bboxes_test.shape[0]:
        rest_idx = set([i for i in range(len(bboxes_test))])
        detected_idx_set = set()
        for idx in detected_anno:
            detected_idx_set.add(idx[0])
        rest_idx = rest_idx - detected_idx_set
        rest_bboxes = [bboxes_test[idx] for idx in rest_idx]
        rest_labels = [labels_test[idx] for idx in rest_idx]
        for bbox, label in zip(rest_bboxes, rest_labels):
            t.append({
                'box_mean': None,
                'label_mean': None,
                'box_test': bbox.tolist(),
                'label_test': label.item(),
                'err': ERR_BBOX_UNNES,
            })

    return t


def pair_targets(targets):
    bboxes = [target['boxes'] for target in targets]
    labels = [target['labels'] for target in targets]
    bboxes_sets = []
    used_bboxes = []

    def check_pair_in_sets(pair):
        if pair[1] in used_bboxes:
            return
        for bboxes_set in bboxes_sets:
            if pair[1] in bboxes_set:
                return
            if pair[0] in bboxes_set:
                bboxes_set.append(pair[1])
                return
        bboxes_sets.append(pair)
        used_bboxes.append(pair[1])
        return

    for i in range(len(bboxes)-1):
        for j in range(i+1, len(bboxes)):
            overlaps = compute_IoU(bboxes[i], bboxes[j])
            for k in range(overlaps.shape[0]):
                assigned_anno_idx = np.argmax(overlaps, axis=1)
                max_overlap = np.max(overlaps[k, assigned_anno_idx])
                if max_overlap < IOU_TRESHOLD:
                    continue
                new_pair = [{'box': k,
                            'targets_no': i},
                            {'box': assigned_anno_idx[k],
                            'targets_no': j}]
                check_pair_in_sets(new_pair)

    bboxes_ret = []
    labels_ret = []
    for i, bboxes_set in enumerate(bboxes_sets):
        bboxes_ret.append([])
        labels_ret.append([])
        for bbox in bboxes_set:
            bboxes_ret[i].append(bboxes[bbox['targets_no']][bbox['box']])
            labels_ret[i].append(labels[bbox['targets_no']][bbox['box']])
    return bboxes_ret, labels_ret


def nms(boxes, scores, threshold):
    if len(boxes) == 0:
        return

    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]

    areas = (xmax - xmin) * (ymax - ymin)
    order = np.argsort(scores)

    keep = []

    while len(order) > 0:
        idx = order[-1]
        keep.append(boxes[idx])
        order = order[:-1]

        if len(order) == 9:
            break

        xxmin = np.maximum(xmin[idx], xmin[order])
        yymin = np.maximum(ymin[idx], ymin[order])
        xxmax = np.minimum(xmax[idx], xmax[order])
        yymax = np.minimum(ymax[idx], ymax[order])

        w = np.maximum(0.0, xxmax - xxmin + 1)
        h = np.maximum(0.0, yymax - yymin + 1)
        intersection = w*h
        iou = intersection/(areas[idx]+areas[order]-intersection)

        left = np.where(iou < threshold)
        order = order[left]
    return keep 


def average_bboxes(bboxes_sets):
    avg_bboxes = []
    for bboxes_set in bboxes_sets:
        avg_bbox = np.zeros((1, 4), dtype=np.float32)
        for bbox in bboxes_set:
            avg_bbox += bbox
        avg_bboxes.append(avg_bbox / len(bboxes_set))
    bboxes = np.stack(avg_bboxes).astype(np.float32) if len(avg_bboxes) > 0 else np.array(avg_bboxes) 
    return bboxes


def average_classes(classes_sets):
    avg_classes = []
    for classes_set in classes_sets:
        avg_classes.append(np.bincount(classes_set).argmax())
    return np.array(avg_classes, dtype=np.int64)
