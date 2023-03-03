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

    iw = np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0])
    ih = np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + area - iw * ih

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
                'err': ERR_BBOX_UNNES,
            })

    return t


def nms(boxes, labels, scores, threshold, func=None):
    if len(boxes) == 0:
        return np.array([]), np.array([])

    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]

    areas = (xmax - xmin) * (ymax - ymin)
    order = np.argsort(scores)

    keep_boxes = []
    keep_labels = []

    while len(order) > 0:
        idx = order[-1]
        if len(order) == 0:
            break

        order = order[:-1]

        xxmin = np.maximum(xmin[idx], xmin[order])
        yymin = np.maximum(ymin[idx], ymin[order])
        xxmax = np.minimum(xmax[idx], xmax[order])
        yymax = np.minimum(ymax[idx], ymax[order])

        w = np.maximum(0.0, xxmax - xxmin + 1)
        h = np.maximum(0.0, yymax - yymin + 1)
        intersection = w*h
        iou = intersection/(areas[idx]+areas[order]-intersection)

        if func is not None:
            to_mean = order[np.where(iou >= threshold)]
            box_keep = func(to_mean,
                            xmin,
                            ymin,
                            xmax,
                            ymax,
                            labels,
                            scores,
                            idx)
            keep_boxes.append(box_keep)
        else:
            keep_boxes.append(boxes[idx])
        keep_labels.append(labels[idx])
        order = order[np.where(iou < threshold)]
    return np.array(keep_boxes), np.array(keep_labels)


def mean_bbox(mean_idx, xmin, ymin, xmax, ymax, labels, scores, idx):

    weights = np.array([scores[idx]])
    xmin_m = np.array([xmin[idx]])
    ymin_m = np.array([ymin[idx]])
    xmax_m = np.array([xmax[idx]])
    ymax_m = np.array([ymax[idx]])

    for _idx in mean_idx:
        if labels[_idx] == labels[idx]:
            weights = np.append(weights, scores[_idx])
            xmin_m = np.append(xmin_m, xmin[_idx])
            ymin_m = np.append(ymin_m, ymin[_idx])
            xmax_m = np.append(xmax_m, xmax[_idx])
            ymax_m = np.append(ymax_m, ymax[_idx])

    mean_xmin = round(np.average(xmin_m, weights=weights), 2)
    mean_ymin = round(np.average(ymin_m, weights=weights), 2)
    mean_xmax = round(np.average(xmax_m, weights=weights), 2)
    mean_ymax = round(np.average(ymax_m, weights=weights), 2)

    return np.array([mean_xmin, mean_ymin, mean_xmax, mean_ymax])


if __name__ == "__main__":
    boxes = np.array([[10.0, 15.0, 25.0, 60.0],
                      [11.0, 13.0, 25.0, 55.0],
                      [10.0, 20.0, 40.0, 65.0],
                      [90.0, 80.0, 180.0, 150.0], ])
    scores = np.array([0.9, 0.55, 0.5, 0.6])
    labels = np.array([1, 1, 1, 1])
    nms_res, _ = nms(boxes, labels, scores, 0.3)
