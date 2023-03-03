import numpy as np


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


def nms(boxes, labels, scores, threshold, func=mean_bbox):
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