import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from icecream import ic

from utils import reader
from dataloader import LoadOnehot


def load_data(data_path, train_potion=0.8, rand_neg=False, batch_size=32, num_cpu=0):
    """
    Load all data
    :param data_path: Path to txt file contain promoter (1 DNA promoter on 1 line)
    :param train_potion: The potion of dataset spend for training
    :param rand_neg: Add random of DNA to negative datset
    :param batch_size: Batch size for loader
    :param num_cpu: Number of CPU perform load data in prallel
    :return: List of train, val, test dataset for positive and negative datset
    """
    # get dataset
    manual_seed = torch.Generator().manual_seed(42)
    pos_data = LoadOnehot(data_path)
    neg_data = LoadOnehot(data_path, is_pos=False, fake=2)

    # add random dataset to negative dataset
    if rand_neg:
        neg_data_rand = LoadOnehot(data_path, is_pos=False, fake=1)
        neg_data = ConcatDataset([neg_data, neg_data_rand])

    # calculate the size of train and test dataset
    train_num = int(len(pos_data)*train_potion)
    val_num = int(len(pos_data)*(1-train_potion)*0.5)
    split_size = [train_num, val_num, len(pos_data) - train_num - val_num]

    # split dataset
    train_pos, val_pos, test_pos = random_split(pos_data, split_size, generator=manual_seed)
    train_neg, val_neg, test_neg = random_split(neg_data, split_size, generator=manual_seed)

    # data loader
    stack_dataset = [train_pos, val_pos, test_pos, train_neg, val_neg, test_neg]
    stack_loaders = list()
    for dataset in stack_dataset:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_cpu)
        stack_loaders.append(data_loader)

    return stack_loaders


if __name__ == '__main__':
    path = "./data/human/nonTATA/hs_pos_nonTATA.txt"
    out = load_data(path)
    ic(out[0])


