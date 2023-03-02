import json
import os
import fnmatch
import numpy as np


class JsonDataSet():

    def __init__(self, json_dir, filename_format='output_res_') -> None:
        self.json_dir = json_dir
        self.filename_format = filename_format

    def __getitem__(self, i):
        bboxes_mean = []
        bboxes_test = []
        labels_mean = []
        labels_test = []
        errors = []
        with open(os.path.join(self.json_dir,
                               f'{self.filename_format}{i}.json')) as f:
            for line in f:
                res = json.loads(line)
                self._append_bbox(res['box_mean'], bboxes_mean)
                self._append_bbox(res['box_test'], bboxes_test)
                self._append_label(res['label_mean'], labels_mean)
                self._append_label(res['label_test'], labels_test)
                errors.append(res['err'])
        try:
            bboxes_mean = np.stack(bboxes_mean).astype(np.float32)
            bboxes_test = np.stack(bboxes_test).astype(np.float32)
            labels_mean = np.stack(labels_mean).astype(np.int32)
            labels_test = np.stack(labels_test).astype(np.int32)
            errors = np.stack(errors).astype(np.int32)
            errors_sorted = np.argsort(-errors)
        except ValueError:
            return {'boxes_mean': [],
                    'labels_mean': [],
                    'boxes_test': [],
                    'labels_test': [],
                    'errs': []}
        return {'boxes_mean': bboxes_mean[errors_sorted],
                'labels_mean': labels_mean[errors_sorted],
                'boxes_test': bboxes_test[errors_sorted],
                'labels_test': labels_test[errors_sorted],
                'errs': errors[errors_sorted]}

    def __len__(self):
        return len(fnmatch.filter(os.listdir(self.json_dir),
                                  f'{self.filename_format}*'))

    def _append_bbox(self, bbox, arr):
        if bbox is not None:
            if type(bbox[0]) == list:
                arr.append(bbox[0])
            else:
                arr.append(bbox)
        else:
            arr.append([0, 0, 0, 0])

    def _append_label(self, label, arr):
        if label is not None:
            arr.append(label)
        else:
            arr.append(-1)
