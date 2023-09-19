import os
from xml_script import get_objects, add_attrs
from tqdm import tqdm


VOC_DIR = './VOCdevkit/VOC2007'


def get_img_ids(filename):
    with open(filename) as file:
        images = [item.strip().split()[0] for item in file]
    return images


def cmp_label(obj1, obj2):
    return obj1.find('name').text == obj2.find('name').text


def cmp_bndbox(obj1, obj2):
    for kword in ("xmin", "ymin", "xmax", "ymax"):
        if obj1.find('bndbox').find(kword).text != obj2.find('bndbox').find(kword).text:
            return False
    return True


def set_renamed(objs_ref, objs_mod):
    for obj_ref, obj_mod in zip(objs_ref, objs_mod):
        if not cmp_label(obj_ref, obj_mod):
            obj_ref.find('renamed').text = '1'


def set_resized(objs_ref, objs_mod):
    for obj_ref, obj_mod in zip(objs_ref, objs_mod):
        if not cmp_bndbox(obj_ref, obj_mod):
            obj_ref.find('resized').text = '1'


def set_removed(objs_ref, objs_mod):
    found = False
    for obj_ref in objs_ref:
        for obj_mod in objs_mod:
            if cmp_bndbox(obj_mod, obj_ref) and cmp_label(obj_mod, obj_ref):
                found = True
                break
        if not found:
            obj_ref.find('removed').text = '1'


if __name__ == "__main__":
    items = {
             'rename': set_renamed,
             'resize': set_resized,
             'remove': set_removed, }
    for key in items.keys():
        imgs_ids = get_img_ids(os.path.join(VOC_DIR, 'ImageSets', 'Main', f'test_{key}.txt'))
        for id in tqdm(imgs_ids, f'Adding atributes to {key}'):
            ref_objs, ref_tree = get_objects(os.path.join(VOC_DIR, "ReferAnno", f"{id}.xml"))
            mod_objs, _ = get_objects(os.path.join(VOC_DIR, 'Annotations', f'{id}.xml'))
            for obj in ref_objs:
                add_attrs(None, obj)
            items[key](ref_objs, mod_objs)
            ref_tree.write(os.path.join(VOC_DIR, "ReferAnno", f"{id}.xml"))
