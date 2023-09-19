import os
from random import sample

DATA_DIR = './VOCdevkit/VOC2007/'

def split_set():
    img_sets = os.path.join(DATA_DIR, "ImageSets", "Main")
    train_ids = [id.strip() for id in open(os.path.join(img_sets, "test.txt"))]
    num_instances = 500
    train_ids = set(train_ids)
    splits = ["remove", "resize", "rename"]
    for name in splits:
        subset = set(sample(train_ids, k=num_instances))
        train_ids = train_ids.difference(subset)
        with open(os.path.join(img_sets, f"test_{name}.txt"), 'a') as f:
            for item in subset:
                f.write(f"{item}\n")
    # with open(os.path.join(img_sets, f"test.txt"), 'w') as f:
    #     for item in train_ids:
    #         f.write(f"{item}\n")

if __name__ == "__main__":
    split_set()
