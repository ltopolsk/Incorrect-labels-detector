import os
import xml.etree.ElementTree as ET
from .voc_dataset import VOC_BBOX_LABEL_NAMES
import numpy as np


class ReferVOCDataset():

    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False,):
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __getitem__(self, i):
        id_ = self.ids[i]
        tree = ET.parse(
            os.path.join(self.data_dir, 'ReferAnno', id_ + '.xml'))
        bbox = []
        label = []
        difficult = []
        renamed = []
        resized = []
        removed = []
        for obj in tree.findall('object'):
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue
            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
            renamed_single =obj.find('renamed').text if obj.find('renamed') is not None else 0
            resized_single =obj.find('resized').text if obj.find('resized') is not None else 0
            removed_single =obj.find('removed').text if obj.find('removed') is not None else 0


            renamed_single = int(renamed_single)
            resized_single = int(resized_single)
            removed_single = int(removed_single)
            renamed.append(renamed_single)
            resized.append(resized_single)
            removed.append(removed_single)
        try:
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
        except ValueError:
            pass
        difficult = np.array(difficult, dtype=np.bool_).astype(np.uint8)
        renamed = np.array(renamed, dtype=np.bool_).astype(np.uint8)
        resized = np.array(resized, dtype=np.bool_).astype(np.uint8)
        removed = np.array(removed, dtype=np.bool_).astype(np.uint8)
        sort_indicies = np.flip(np.argsort(label))
        return {'boxes': bbox[sort_indicies],
                'labels': label[sort_indicies],
                'difficult': difficult[sort_indicies],
                'renamed': renamed[sort_indicies],
                'resized': resized[sort_indicies],
                'removed': removed[sort_indicies], }

    def __len__(self):
        return len(self.ids)
