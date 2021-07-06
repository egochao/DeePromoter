import random
import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader, random_split

from utils import neggen, reader, get_list_kmer, protein2num


class LoadOnehot(Dataset):
    def __init__(self, pathpos, is_pos=True, device="cuda", fake=0, length_pro=300, divide=20, part=8):
        if is_pos and fake != 0:
            raise Exception("Cant use key word fake on positive dataset")
        self.device = device
        self.fake = fake
        self.length_pro = length_pro
        self.divide = divide
        self.part = part

        # get list of kmer
        dic = get_list_kmer(1)

        # read data from file
        self.dpos = reader(pathpos)

        # convert protein to number sequence
        self.npos = [protein2num(pro, dic) for pro in self.dpos]

        if is_pos:
            self.poslabel = torch.from_numpy(np.ones(len(self.dpos)))
        else:
            # ic("go false")
            self.poslabel = torch.from_numpy(np.zeros(len(self.dpos)))
        self.poslabel = self.poslabel.to(device)

    def __len__(self):
        return len(self.dpos)

    def __getitem__(self, idx):
        # convert data to one hot format and up to device
        pro = self.npos[idx]
        if len(pro) < self.length_pro:
            pro = pro + [0] * (self.length_pro - len(pro))
        elif len(pro) > self.length_pro:
            pro = pro[:self.length_pro]

        # random generate a fake promoter by shuffle the pro
        if self.fake == 1:
            pro = random.shuffle(pro)
        # random generate a fake promoter by replace part of pro
        elif self.fake == 2:
            pro = neggen(pro, num_part=self.divide, keep=self.part, max_class=4)

        torchpro = torch.from_numpy(np.array(pro))
        onehot = torch.nn.functional.one_hot(torchpro, num_classes=4).to(self.device)
        return onehot.float(), self.poslabel[idx]


