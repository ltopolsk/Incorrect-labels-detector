import os
import json
import torch as t
import numpy as np
from compare_module.compare import compare
import compare_module.config as c
from compare_module.nms import nms
from verify_funcs.visualize_funcs import cust_vis_bbox
from verify_funcs.utils import (NUM_SETS, DATA_DIR, MODELS_DIR, OUTPUT_DIR)
from model import FasterRCNNVGG16
from utils.vis_tool import vis_image
from utils import array_tool as at
from data.dataset import TestDataset
from torch.utils.data import DataLoader

errs = {
    c.TRUE_POS: 0,
    c.ERR_BBOX_SIZE: 0,
    c.ERR_BBOX_UNNES: 0,
    c.ERR_LABEL: 0,
    c.ERR_LACK_BBOX: 0,
    c.ERR_REDUN: 0,
}


def read_models():
    # models = [FasterRCNNVGG16().cuda().load_state_dict(t.load(os.path.join(MODELS_DIR, f'model_{i+1}'))['model']) for i in range(NUM_SETS)]
    models = []
    for i in range(NUM_SETS):
        faster_rcnn = FasterRCNNVGG16()
        state_dict = t.load(os.path.join(MODELS_DIR, f'model_{i+1}'))
        faster_rcnn.load_state_dict(state_dict['model'])
        models.append(faster_rcnn.cuda())
    return models


def save_cmp_results(cmp_res, id):
    with open(os.path.join(OUTPUT_DIR, 'res', f'output_res_{id}.json'), 'w') as f:
        for res in cmp_res:
            json.dump(res, f)
            f.write('\n')


def compare_labels_single(img, test_bboxes, test_labels, trainers):
    boxes, labels, scores = np.array([]), np.array([]), np.array([])
    for model in trainers:
        _bboxes, _labels, _scores = model.predict(img, visualize=True)
        np.append(boxes, _bboxes[0])
        np.append(labels, _labels[0])
        np.append(scores, _scores[0])
    avg_bboxes, avg_labels = nms(boxes=t.tensor(boxes),
                                 labels=t.tensor(labels),
                                 scores=t.tensor(scores),
                                 threshold=c.IOU_TRESHOLD,
                                 func=c.FUNC)
    return compare(avg_bboxes, avg_labels, test_bboxes, test_labels)


def compare_labels_ds(dataloader, trainers):
    global errs
    for i, (img, _, bboxes, labels, _) in enumerate(dataloader):
        img, bboxes, labels = img.cuda().squeeze(0), bboxes.cuda().squeeze(0), labels.cuda().squeeze(0)
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
    models = read_models()
    ds = TestDataset(voc_data_dir=DATA_DIR)
    data_loader = DataLoader(dataset=ds,
                             batch_size=1,
                             shuffle=False,
                             num_workers=2,
                             pin_memory=True)
    compare_labels_ds(data_loader, models)
    with open(os.path.join(OUTPUT_DIR, 'res', 'output_errs.json'), 'w') as f:
        json.dump(errs, f)
        f.write('\n')
