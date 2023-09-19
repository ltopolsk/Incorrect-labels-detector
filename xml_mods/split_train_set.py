import os
from random import sample


def split_set(data_dir, num_sets):
    train_ids = set([id.strip() for id in open(os.path.join(data_dir,"ImageSets", "Main","val.txt"))])
    num_instances = int(len(train_ids)/num_sets)
    for i in range(num_sets):
        cur_set = set(sample(train_ids, num_instances))
        train_ids -= cur_set
        with open(os.path.join(data_dir,"ImageSets", "Main", f"val_5_{i+1}.txt"), 'a') as f:
            for item in cur_set:
                f.write(f"{item}\n")
