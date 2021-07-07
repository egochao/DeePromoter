import torch
import torch.optim as optim
from icecream import ic
from pathlib import Path

from dataloader import load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    path = "./data/human/nonTATA/hs_pos_nonTATA.txt"
    out = load_data(path)
    ic(out[0])


