import os
import json
import torch as t
from compare_module.compare import compare
import compare_module.utils as c
from compare_module.nms import nms, mean_bbox
from verify_funcs.visualize_funcs import cust_vis_bbox
from utils.config import opt
from model import FasterRCNNVGG16
from utils.vis_tool import vis_image
from utils import array_tool as at
from data.dataset import VOCBboxDataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

errs = {
    c.TRUE_POS: 0,
    c.ERR_BBOX_SIZE: 0,
    c.ERR_BBOX_UNNES: 0,
    c.ERR_LABEL: 0,
    c.ERR_LACK_BBOX: 0,
    # c.ERR_REDUN: 0,
}

names = {
    c.TRUE_POS: "TP", 
    c.ERR_LABEL: "ERR_CLASS",
    c.ERR_BBOX_SIZE: "ERR_BBOX_SIZE",
    c.ERR_LACK_BBOX: "ERR_MISSING",
    c.ERR_BBOX_UNNES: "ERR_UNNESS"
}

def read_models():
    models = []
    for i in range(opt.num_sets):
        faster_rcnn = FasterRCNNVGG16()
        state_dict = t.load(os.path.join(opt.models_dir, f'model_train_{i+1}'))
        faster_rcnn.load_state_dict(state_dict['model'])
        models.append(faster_rcnn.cuda())
    return models


def save_cmp_results(cmp_res, id):
    with open(os.path.join(opt.output_dir, 'res', f'output_res_{id}.json'), 'w') as f:
        for res in cmp_res:
            json.dump(res, f)
            f.write('\n')


def compare_labels_single(img, test_bboxes, test_labels, trainers):
    boxes, labels, scores = [], [], []
    for model in trainers:
        _bboxes, _labels, _scores = model.predict(img, visualize=True)
        boxes.extend(_bboxes[0])        
        labels.extend(_labels[0])
        scores.extend(_scores[0])
    f = mean_bbox if opt.use_mean else None
    avg_bboxes, avg_labels, avg_scores = nms(boxes=t.tensor(np.array(boxes)),
                                             labels=t.tensor(np.array(labels)),
                                             scores=t.tensor(np.array(scores)),
                                             threshold=0.7,
                                             func=f)
    return compare(avg_bboxes, avg_labels, test_bboxes, test_labels, avg_scores, opt.iou)


def compare_labels_ds(dataloader, trainers):
    global errs
    it = tqdm(dataloader, desc='Processing data')
    for i, (img, bboxes, labels, _) in enumerate(it):
        img, bboxes, labels = img.cuda(), bboxes.cuda().squeeze(0), labels.cuda().squeeze(0)
        cmp_res = compare_labels_single(img, bboxes, labels, trainers)
        for res in cmp_res:
            errs[res['err']] += 1
        if i % 10 == 0 and opt.vis_verify_res:
            vis_result(img, cmp_res, f'output_vis_{i}.png')
        save_cmp_results(cmp_res, i)


def vis_result(img, cmp_res, filename):
    groups = {err_key: [] for err_key in errs.keys()}
    for res in cmp_res:
        groups[res["err"]].append(res)
    
    for item, val in groups.items():
        if len(val)>0: 
            ax = vis_image(at.tonumpy(img[0]))
            cust_vis_bbox(ax, val, os.path.join(opt.output_dir, 'vis', names[item],filename))


def verify(**kwargs):
    opt._parse(kwargs)
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
        os.makedirs(os.path.join(opt.output_dir, 'res'))
        os.makedirs(os.path.join(opt.output_dir, 'vis'))
    models = read_models()
    ds = VOCBboxDataset(data_dir=opt.voc_data_dir, split='test')
    data_loader = DataLoader(dataset=ds,
                             batch_size=1,
                             shuffle=False,
                             num_workers=8,
                             pin_memory=True)
    if opt.vis_verify_res:
        for err_type in ("TP", "ERR_CLASS","ERR_BBOX_SIZE", "ERR_MISSING", "ERR_UNNESS"):
            if not os.path.exists(os.path.join(opt.output_dir, 'vis', err_type)):
                os.makedirs(os.path.join(opt.output_dir, 'vis', err_type))
    compare_labels_ds(data_loader, models)
    with open(os.path.join(opt.output_dir, 'res', 'output_errs.json'), 'w') as f:
        json.dump(errs, f)
        f.write('\n')

if __name__ == "__main__":
    verify()