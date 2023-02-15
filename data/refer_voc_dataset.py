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
            renamed.append(int(obj.find('renamed').text))
            resized.append(int(obj.find('resized').text))
            removed.append(int(obj.find('removed').text))

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool_).astype(np.uint8)
        renamed = np.array(renamed, dtype=np.bool_).astype(np.uint8)
        resized = np.array(resized, dtype=np.bool_).astype(np.uint8)
        removed = np.array(removed, dtype=np.bool_).astype(np.uint8)
        return {'boxes': bbox,
                'labels': label,
                'difficult': difficult,
                'renamed': renamed,
                'resized': resized,
                'removed': removed, }

    def __len__(self):
        return len(self.ids)
