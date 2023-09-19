import torch as t


def mean_bbox(mean_idx, xmin, ymin, xmax, ymax, labels, scores, idx):

    weights = t.tensor([scores[idx]]).cuda()
    xmin_m = t.tensor([xmin[idx]]).cuda()
    ymin_m = t.tensor([ymin[idx]]).cuda()
    xmax_m = t.tensor([xmax[idx]]).cuda()
    ymax_m = t.tensor([ymax[idx]]).cuda()

    for _idx in mean_idx:
        if labels[_idx] == labels[idx]:
            weights = t.cat((weights, scores[_idx].unsqueeze(0)))
            xmin_m = t.cat((xmin_m, xmin[_idx].unsqueeze(0)))
            ymin_m = t.cat((ymin_m, ymin[_idx].unsqueeze(0)))
            xmax_m = t.cat((xmax_m, xmax[_idx].unsqueeze(0)))
            ymax_m = t.cat((ymax_m, ymax[_idx].unsqueeze(0)))

    mean_xmin = t.unsqueeze(xmin_m@weights/weights.sum(), 0)
    mean_ymin = t.unsqueeze(ymin_m@weights/weights.sum(), 0)
    mean_xmax = t.unsqueeze(xmax_m@weights/weights.sum(), 0)
    mean_ymax = t.unsqueeze(ymax_m@weights/weights.sum(), 0)

    return t.round(t.cat([mean_xmin, mean_ymin, mean_xmax, mean_ymax]), decimals=2).unsqueeze(0), labels[idx].unsqueeze(0), weights.mean().unsqueeze(0)


def nms(boxes, labels, scores, threshold, func=mean_bbox):
    if boxes.shape[0] == 0:
        return t.empty((0, 4)), t.empty((0)), t.empty((0))

    boxes, labels, scores = boxes.cuda(), labels.cuda(), scores.cuda()

    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]

    areas = (xmax - xmin) * (ymax - ymin)
    order = t.argsort(scores).cuda()

    keep_boxes = t.empty((0, 4)).cuda()
    keep_labels = t.empty((0)).cuda()
    keep_scores = t.empty((0)).cuda()

    while len(order) > 0:
        idx = order[-1]
        if len(order) == 0:
            break

        order = order[:-1]

        xxmin = t.maximum(xmin[idx], xmin[order])
        yymin = t.maximum(ymin[idx], ymin[order])
        xxmax = t.minimum(xmax[idx], xmax[order])
        yymax = t.minimum(ymax[idx], ymax[order])

        w = t.clamp(xxmax - xxmin + 1, min=0.0)
        h = t.clamp(yymax - yymin + 1, min=0.0)
        intersection = w*h
        iou = intersection/(areas[idx]+areas[order]-intersection)

        if func is not None:
            to_mean = order[t.nonzero(order.where(iou >= threshold, t.zeros(size=order.shape, dtype=bool).cuda()), as_tuple=True)]
            box_keep, label_keep, score_keep = func(to_mean, xmin, ymin, xmax, ymax, labels, scores, idx)
            keep_boxes = t.cat((keep_boxes, box_keep))
            keep_labels = t.cat((keep_labels, label_keep))
            keep_scores = t.cat((keep_scores, score_keep))
        else:
            keep_boxes = t.cat((keep_boxes, boxes[idx].unsqueeze(0)), dim=0)
            keep_labels = t.cat((keep_labels, labels[idx].unsqueeze(0)))
            keep_scores = t.cat((keep_scores, scores[idx].unsqueeze(0)))
        order = order[t.nonzero(order.where(iou < threshold, t.zeros(size=order.shape, dtype=bool).cuda()), as_tuple=True)]
    return keep_boxes, keep_labels, keep_scores


if __name__ == '__main__':
    boxes = t.tensor([[10.0, 15.0, 25.0, 60.0],
                      [11.0, 13.0, 25.0, 55.0],
                      [10.0, 20.0, 40.0, 65.0],
                      [20.0, 5.0, 27.0, 60.0], ])
    scores = t.tensor([0.9, 0.7, 0.5, 0.1])
    labels = t.tensor([1, 1, 1, 1])
    nms_res, _ = nms(boxes, labels, scores, 0.3, func=None)
