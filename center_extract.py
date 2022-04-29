import argparse
import os
import torch
import cv2
import joblib
import pickle
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
from models.mem_cvae import HFVAD
from datasets.dataset import Chunked_sample_dataset
from utils.eval_utils import save_evaluation_curves,evaluation



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", type=str,
                        default="./pretrained_ckpts/ped2_HF2VAD_99.31.pth",
                        help='path to pretrained weights')
    parser.add_argument("--cfg_file", type=str,
                        default="./pretrained_ckpts/ped2_HF2VAD_99.31_cfg.yaml",
                        help='path to pretrained model configs')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.cfg_file))
    testing_chunked_samples_file = os.path.join("./data", config["dataset_name"],
                                                "testing/chunked_samples/chunked_samples_00.pkl")

    from train import cal_training_condi_center

    os.makedirs(os.path.join("./eval", config["exp_name"]), exist_ok=True)
    training_chunked_samples_dir = os.path.join("./data", config["dataset_name"], "training/chunked_samples")
    training_center_path = os.path.join("./eval", config["exp_name"], "training_centers.npy")
    cal_training_condi_center(config, args.model_save_path, training_chunked_samples_dir, training_center_path)

