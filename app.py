import os
import argparse
from utils.config import opt
from train import train
from xml_mods.split_train_set import split_set
from verify_app import verify
from compute_stats import compute

def train_models(**kwargs):
    opt._parse(kwargs)
    split_set(data_dir=opt.voc_data_dir, num_sets=opt.num_sets)
    for i in range(opt.num_sets):
        train(split=f'train_{i+1}')

def verify_dataset(**kwargs):
    opt._parse(kwargs)
    verify()

def compute_stats(**kwargs):
    opt._parse(kwargs)
    compute()

if __name__=="__main__":
    import fire
    fire.Fire()