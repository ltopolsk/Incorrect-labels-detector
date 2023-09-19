import argparse
import xml.etree.ElementTree as ET
import os
from random import randint, choice
import numpy as np


def rem(root, obj):
    root.remove(obj)


def rename(root, obj):
    cur_name = obj.find('name').text
    temp = NAMES.copy()
    temp.remove(cur_name)
    obj.find('name').text = choice(temp)


def add_attrs(root, obj):
    resized = ET.SubElement(obj, 'resized')
    renamed = ET.SubElement(obj, 'renamed')
    removed = ET.SubElement(obj, 'removed')
    resized.text = '0'
    renamed.text = '0'
    removed.text = '0'


def get_size(root):
    size = {
        "width": int(root.find('size').find('width').text),
        "height": int(root.find('size').find('height').text),
    }
    return size


def fit_num(value, measurement, size):
    if value > size[measurement]:
        new_value = size[measurement]
    elif value < 1:
        new_value = 1
    else:
        new_value = value
    return new_value


def resize_bbox(root, obj):
    size = get_size(root)
    bbox = obj.find('bndbox')

    new_bbox_vals = {
        "xmin": int(np.random.normal(int(bbox.find("xmin").text), 50)),
        "xmax": int(np.random.normal(int(bbox.find("xmax").text), 50)),
        "ymin": int(np.random.normal(int(bbox.find("ymin").text), 50)),
        "ymax": int(np.random.normal(int(bbox.find("ymax").text), 50)),
    }
    if new_bbox_vals["xmin"] > new_bbox_vals["xmax"]:
        new_bbox_vals["xmin"], new_bbox_vals["xmax"] = new_bbox_vals["xmax"], new_bbox_vals["xmin"]
    if new_bbox_vals["ymin"] > new_bbox_vals["ymax"]:
        new_bbox_vals["ymin"], new_bbox_vals["ymax"] = new_bbox_vals["ymax"], new_bbox_vals["ymin"]

    if new_bbox_vals["xmin"] == new_bbox_vals["xmax"]:
        if new_bbox_vals["xmin"] > size['width']/2:
            new_bbox_vals["xmin"] -= 10
        else:
            new_bbox_vals["xmax"] += 10
    if new_bbox_vals["ymin"] == new_bbox_vals["ymax"]:
        if new_bbox_vals["ymin"] > size['height']/2:
            new_bbox_vals["ymin"] -= 10
        else:
            new_bbox_vals["ymax"] += 10
    fit_num(new_bbox_vals["xmin"], 'width', size)
    fit_num(new_bbox_vals["xmax"], 'width', size)
    fit_num(new_bbox_vals["ymin"], 'height', size)
    fit_num(new_bbox_vals["ymax"], 'height', size)
    for item in new_bbox_vals.keys():
        bbox.find(item).text = str(new_bbox_vals[item])


def modify_objects(filepath, prob, total_modified, total_num_obj, func, cat=None):
    tree = ET.parse(filepath)
    root = tree.getroot()
    modified = 0
    for obj in root.iter('object'):
        if (func == 'rem' and root_num_objs(root) == 1):
            continue
        if obj.find('difficult').text == '0' and (cat is None or obj.find('name').text == cat) and randint(0, 100) < prob*100:
            globals()[func](root, obj)
            modified += 1
            if abs(prob - (total_modified + modified)/total_num_obj) <= DELTA:
                break
    tree.write(filepath)
    return modified


def root_num_objs(root, cat=None):
    to_ret = 0
    for obj in root.iter('object'):
        if obj.find('difficult').text == '0' and (cat is None or obj.find('name').text == cat):
            to_ret += 1
    return to_ret


def num_objects(filepath, cat=None):
    tree = ET.parse(filepath)
    root = tree.getroot()
    return root_num_objs(root, cat)


def get_objects(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    return root.findall('object'), tree


def get_names():
    images = get_img_ids(SPLIT)
    names = set([])
    for img in images:
        objects, _ = get_objects(os.path.join(VOC_DIR, f"Annotations/{img}.xml"))
        for obj in objects:
            names.add(obj.find('name').text)
    return list(names)


def get_img_ids(filename):
    with open(os.path.join(VOC_DIR, f"ImageSets/Main/{filename}.txt")) as file:
        images = [item.strip().split()[0] for item in file]
    return images


def count_total_obj(img_ids, cat=None):
    total_num_obj = 0
    for img in img_ids:
        total_num_obj += num_objects(os.path.join(VOC_DIR, f"Annotations/{img}.xml"), cat)
    return total_num_obj


def do_operation(num, func, filename, cat=None):
    images = get_img_ids(filename)
    total_num_obj = count_total_obj(images, cat)
    total_modified = 0
    cat_print = cat if cat is not None else "_"
    print(f"loop {0}. ({(cat_print)})")
    for img in images:
        total_modified += modify_objects(os.path.join(VOC_DIR, f"Annotations/{img}.xml"), num, total_modified, total_num_obj, func, cat)
    if func == "rem":
        i = 1
        while num - total_modified/total_num_obj > DELTA and i < 10:
            print(f"loop {i}. ({cat})")
            i += 1
            for img in images:
                total_modified += modify_objects(os.path.join(VOC_DIR, f"Annotations/{img}.xml"), num, total_modified, total_num_obj, func, cat)
            if abs(num - total_modified/total_num_obj) <= DELTA:
                break
    return total_modified, total_num_obj


def do_rem():
    total_num_obj = count_total_obj(get_img_ids(SPLIT))
    total_removed = 0
    for i, cat in enumerate(NAMES):
        print(f"class {i}/{len(NAMES)}")
        removed, num_obj = do_operation(PER, FUNC, f"{cat.lower()}_{SPLIT}".replace("_remove", ""), cat)
        print(f"CLASS {cat}\nremoved: {removed}, total: {num_obj}\n{removed/num_obj:.04f}")
        total_removed += removed
    print(f"Total objs removed: {total_removed}\n{total_removed/total_num_obj:.04f}")


def do_default():
    total_removed, total_num_obj = do_operation(PER, FUNC, SPLIT)
    print(f"Total objs modified: {total_removed}\n{total_removed/total_num_obj:.04f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--function')
    parser.add_argument('-p', '--percent')
    parser.add_argument('-d', '--dir')
    parser.add_argument('-s', '--split')
    return parser.parse_args()


def config_script(args):
    global VOC_DIR
    global DELTA
    global PER
    global FUNC
    global NAMES
    global SPLIT

    VOC_DIR = args.dir
    SPLIT = args.split
    DELTA = 0.005
    PER = float(args.percent) if args.percent else 1.0
    FUNC = args.function.lower()
    NAMES = get_names()


if __name__ == "__main__":    
    args = parse_args()
    config_script(args)
    print(PER)
    # if FUNC == "rem":
        # do_rem()
    # else:
    do_default()
    print('DONE')