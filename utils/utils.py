import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml


def seed_set(SEED):
    if not SEED:
        SEED = np.random.randint(0, 10000)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.use_deterministic_algorithms(True)
    return SEED

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config