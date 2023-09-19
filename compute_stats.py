from compare_module.iou import compute_IoU
from utils.config import opt
from data.json_dataset import JsonDataSet
from data.refer_voc_dataset import ReferVOCDataset
import numpy as np
import os
import torch as t
from tqdm import tqdm
import json



def compute_img_detections(targets_ref, targets_json, stats, roc_stat, stats_v2):

    used_targs_ref = []

    def check_empty(stat_empty, stat_dict):
        if not len(targets_ref['boxes']):
            stats[stat_empty] += 1
            stat_dict[stat_empty] += 1
            return 1

    def compare_test_refer(stats_incr, stat_empty, stat_dict, stat_dict_inc):
        if check_empty(stat_empty, stat_dict): return
        _idx = np.where((targets_ref['boxes'] == targets_json['boxes_test'][i]).all(axis=1))[0]
        roc_stat[targets_json['errs'][i]][0].append(targets_json['scores'][i])
        if len(_idx) != 0:
            idx = _idx[0]
            used_targs_ref.append(idx)

            roc_stat[targets_json['errs'][i]][1].append(int((targets_json['labels_test'][i] == targets_ref['labels'][idx] and stat_empty=='fp')
                                                        or (targets_json['labels_test'][i] != targets_ref['labels'][idx] and stat_empty=='fn')))

            stats[stats_incr[0]] += int(targets_ref['labels'][idx] == targets_json['labels_test'][i])
            stats[stats_incr[1]] += int(targets_ref['labels'][idx] != targets_json['labels_test'][i])
            stat_dict[stat_dict_inc[0]] += int(targets_ref['labels'][idx] == targets_json['labels_test'][i])
            stat_dict[stat_dict_inc[1]] += int(targets_ref['labels'][idx] != targets_json['labels_test'][i])
        else:
            stats[stat_empty] += 1
            stat_dict[stat_empty] += 1
            roc_stat[targets_json['errs'][i]][1].append(0)

    def compare_mean_refer(stats_incr, stat_empty, stat_dict, stat_dict_inc, ref_keyword):
        if check_empty(stat_empty, stat_dict): return
        overlaps = compute_IoU(t.unsqueeze(t.from_numpy(targets_json['boxes_mean'][i]), 0), t.from_numpy(targets_ref['boxes']))
        assigned_anno_idx = t.argmax(overlaps).numpy()
        roc_stat[targets_json['errs'][i]][0].append(targets_json['scores'][i])
        if assigned_anno_idx.tolist() in used_targs_ref:
            stats[stats_incr[1]] += 1
            stat_dict[stat_dict_inc[1]] += 1
            roc_stat[targets_json['errs'][i]][1].append(0)
        else:
            max_overlap = overlaps[assigned_anno_idx]
            if max_overlap >= opt.iou and targets_ref[ref_keyword][assigned_anno_idx]:
                stats[stats_incr[0]] += 1
                stat_dict[stat_dict_inc[0]] += 1
                used_targs_ref.append(assigned_anno_idx.tolist())
                roc_stat[targets_json['errs'][i]][1].append(1)
            else:
                stats[stats_incr[1]] += 1
                stat_dict[stat_dict_inc[1]] += 1
                roc_stat[targets_json['errs'][i]][1].append(0)

    def compare_test_not_ref(stats_incr, stat_empty, stat_dict, stat_dict_inc):
        if check_empty(stat_empty, stat_dict): return
        _idx = np.where((targets_ref['boxes'] == targets_json['boxes_test'][i]).all(axis=1))[0]
        if len(_idx) == 0:
            stats[stats_incr[0]] += 1
            stat_dict[stat_dict_inc[0]] += 1
        else:
            stats[stats_incr[1]] += 1
            stat_dict[stat_dict_inc[1]] += 1
            used_targs_ref.append(_idx[0])

    def get_command_dict():
        return {0: (compare_test_refer, (('tp', 'fp'), 'fp', stats_v2[0], ('tp','fp'))),
                -1: (compare_test_refer, (('fn', 'tn'), 'fn', stats_v2[-1], ('fp', 'tp'))),
                -2: (compare_mean_refer, (('tn', 'fn'), 'fn', stats_v2[-2], ('tp', 'fp'),'resized')),
                -3: (compare_mean_refer, (('tn', 'fn'), 'tn', stats_v2[-3], ('tp', 'fp'),'removed')),
                -4: (compare_test_not_ref, (('tn', 'fn'), 'tn', stats_v2[-4], ('tp','fp'))),}
    
    for i in range(len(targets_json['errs'])):
        command_dict = get_command_dict()
        func, args = command_dict[targets_json['errs'][i]]
        func(*args)

    if len(used_targs_ref) < np.array(targets_ref['boxes']).shape[0]:
        rest_idx = set(i for i in range(targets_ref['boxes'].shape[0])) - set(used_targs_ref)
        for i in rest_idx:
            if targets_ref['renamed'][i] == 1:
                stats_v2[-1]['fn'] += 1
            elif targets_ref['resized'][i] == 1:
                stats_v2[-2]['fn'] += 1
            elif targets_ref['removed'][i] == 1:
                stats_v2[-3]['fn'] += 1
            else:
                stats_v2[0]['fn'] += 1
        stats['fp'] += len(rest_idx)

def compute():
    OUTPUT_DIR = os.path.join(opt.output_dir, 'res')
    VOC_DIR = opt.voc_data_dir
    json_ds = JsonDataSet(OUTPUT_DIR)
    refer_ds = ReferVOCDataset(VOC_DIR, split='test')
    roc_stat = {-i:([],[]) for i in range(4)}
    stats = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    stats_v2 ={
        0 : {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        -1: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        -2: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        -3: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
        -4: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    }
    for i in tqdm(range(len(json_ds)), desc='Computing detections'):
        compute_img_detections(refer_ds[i], json_ds[i], stats, roc_stat, stats_v2)

    metrics = {'acc': ((stats["tp"] + stats["tn"])/(stats["tp"]+stats["tn"]+stats["fp"]+stats["fn"])),
               'prec': (stats["tp"]/(stats["tp"]+stats["fp"])),
               'prec_negative': (stats["tn"]/(stats["tn"]+stats["fn"])),
               'recall': (stats["tp"]/(stats["tp"]+stats["fn"])),
               'recall_negative': (stats["tn"]/(stats["tn"]+stats["fp"])),
               'fpr':(stats["fp"]/(stats["tn"]+stats["fp"])),
               'fnr': (stats["fn"]/(stats["tp"]+stats["fn"])),}
    file_name = '0_' + OUTPUT_DIR.split('/')[-1].rstrip('\\res') + '_' + ('mean_box' if opt.use_mean else 'nms') + '_test'
    if not os.path.exists(os.path.join('.\\confusion-mat', str(opt.num_sets))):
        os.makedirs(os.path.join('.\\confusion-mat', str(opt.num_sets)))
    with open(os.path.join('.\\confusion-mat', str(opt.num_sets), 'metrics', f'{file_name}.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join('.\\confusion-mat', str(opt.num_sets), 'metrics', f'{file_name}_v2.json'), 'w') as f:
        json.dump(stats_v2, f, indent=4)

if __name__ == "__main__":
    compute()
