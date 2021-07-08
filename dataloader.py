import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

from utils import neggen, reader, get_list_kmer, protein2num


class LoadOnehot(Dataset):
    def __init__(self, pathpos, is_pos=True, device="cuda", fake=0, length_pro=300, divide=20, part=8):
        """
        Dataset
        :param pathpos: Path to the txt data file
        :param is_pos: Control the label for dataset True for 1, False for 0
        :param device: Device
        :param fake: 0 for load original txt dataset , 1 for random fake, 2 for faking method as describe in the paper
        :param length_pro: Input sequence length
        :param divide: Number of part to break protein into before replace some part with random sequence
        :param part: Number of part to keep the same when do random subsequence 
        """
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


def load_data(data_path, train_potion=0.8, rand_neg=False, batch_size=32, num_cpu=0, device="cuda"):
    """
    Load all data
    :param data_path: Path to txt file contain promoter (1 DNA promoter on 1 line)
    :param train_potion: The potion of dataset spend for training
    :param rand_neg: Add random of DNA to negative datset
    :param batch_size: Batch size for loader
    :param num_cpu: Number of CPU perform load data in prallel
    :param device: Device to load data on
    :return: List of train, val, test dataset for positive and negative datset
    """
    # get dataset
    manual_seed = torch.Generator().manual_seed(42)
    pos_data = LoadOnehot(data_path, device=device)
    neg_data = LoadOnehot(data_path, is_pos=False, fake=2, device=device)

    # calculate the size of train and test dataset
    train_num = int(len(pos_data)*train_potion)
    val_num = int(len(pos_data)*(1-train_potion)*0.5)
    split_size = [train_num, val_num, len(pos_data) - train_num - val_num]

    # split dataset
    train_pos, val_pos, test_pos = random_split(pos_data, split_size, generator=manual_seed)
    train_neg, val_neg, test_neg = random_split(neg_data, split_size, generator=manual_seed)

    # add random dataset to negative dataset(only to train set)
    if rand_neg:
        neg_data_rand = LoadOnehot(data_path, is_pos=False, fake=1, device=device)
        train_neg = ConcatDataset([train_neg, neg_data_rand])

    # data loader
    stack_dataset = [train_pos, val_pos, test_pos, train_neg, val_neg, test_neg]
    stack_loaders = list()
    for dataset in stack_dataset:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_cpu)
        stack_loaders.append(data_loader)

    return stack_loaders


def load_data_test(data_path, batch_size=32, device="cuda", num_cpu=0):
    dataset = LoadOnehot(data_path, device=device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_cpu)
    return data_loader