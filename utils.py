from torch.utils.tensorboard import SummaryWriter
import os
import string
from datetime import datetime
from matplotlib import pyplot as plt
import torch
import numpy as np
import random


def set_deterministic(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tb_logger():
    vowels = "aeiou"
    consonants = "".join(set(string.ascii_lowercase) - set(vowels))
    run_id = ''.join([f'{random.choice(consonants)}{random.choice(vowels)}' for i in range(4)])
    return SummaryWriter(os.path.join(r'./results/tensorboard', run_id + '-' + datetime.now().strftime('%b%d_%H-%M-%S')))


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    
    if isinstance(obj, list):
        res = []
        for v in obj:
            res.append(to_device(v, device))
        return res
    
    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = to_device(v, device)
        return res
    
    raise TypeError(f'Invalid type ({obj.__class__}) for to_device')
