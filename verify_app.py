import os
import json
import torch as t
import compare_module.compare as cmp
from custom_funcs import cust_vis_bbox
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from utils.vis_tool import vis_image
from utils.config import opt
from utils import array_tool as at
from data.dataset import VOCBboxDataset
import numpy as np

MODELS_DIR = './models/'
DATA_DIR = './VOCdevkit/VOC2007/'
OUTPUT_DIR = './outputs/'
NUM_SETS = 4

errs = {
    cmp.TRUE_POS: 0,
    cmp.ERR_BBOX_SIZE: 0,
    cmp.ERR_BBOX_UNNES: 0,
    cmp.ERR_LABEL: 0,
    cmp.ERR_LACK_BBOX: 0,
    cmp.ERR_REDUN: 0,
}


def read_trainers():
    trainers = []
    faster_rcnn = FasterRCNNVGG16()
    opt.caffe_pretrain = False
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    for i in range(NUM_SETS):
        trainer.load(os.path.join(MODELS_DIR, f'model_{i+1}'))
        trainers.append(trainer)
    return trainers


def read_data(split):
    return VOCBboxDataset(DATA_DIR, split=split)


def save_cmp_results(cmp_res, id):
    with open(os.path.join(OUTPUT_DIR, 'res', f'output_res_{id}.json'), 'w') as f:
        for res in cmp_res:
            json.dump(res, f)
            f.write('\n')


def compare_labels_single(img, test_bboxes, test_labels, trainers):
    boxes, labels, scores = [], [], []
    for trainer in trainers:
        _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
        boxes.extend(_bboxes[0])
        labels.extend(_labels[0])
        scores.extend(_scores[0])
    boxes = np.array(boxes)
    labels = np.array(labels)
    scores = np.array(scores)
    avg_bboxes, avg_labels = cmp.nms(boxes,
                                     labels,
                                     scores,
                                     threshold=0.5,
                                     func=cmp.mean_bbox)
    return cmp.compare(avg_bboxes, avg_labels, test_bboxes, test_labels)


def compare_labels_ds(dataset, trainers):
    global errs
    for i in range(len(dataset)):
        img, bboxes, labels, _ = dataset[i]
        img = t.from_numpy(img)[None].cuda()
        cmp_res = compare_labels_single(img, bboxes, labels, trainers)
        for res in cmp_res:
            errs[res['err']] += 1
        if i % 5 == 1:
            vis_result(img, cmp_res, os.path.join(OUTPUT_DIR, 'vis', f'output_vis_{i}.png'))
        save_cmp_results(cmp_res, i)


def vis_result(img, cmp_res, filename):
    ax = vis_image(at.tonumpy(img[0]))
    cust_vis_bbox(ax, cmp_res, filename)


if __name__ == "__main__":
    trainers = read_trainers()
    ds = read_data('test')
    compare_labels_ds(ds, trainers)
    with open(os.path.join(OUTPUT_DIR, 'res', 'output_errs.json'), 'w') as f:
        json.dump(errs, f)
        f.write('\n')
