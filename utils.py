import itertools
import random
import numpy as np


def reader(filep):
    with open(filep, "r") as f:
        data = f.readlines()
    data = [da.strip() for da in data]
    return data


def get_list_kmer(kmer, string="TGCA"):
    string = list(string)
    lkmer = ["".join(p) for p in itertools.product(string, repeat=kmer)]
    lkmer.sort()
    return lkmer


def protein2num(protein, elements):
    kmer = len(elements[0])
    edict = dict()
    for i, ele in enumerate(elements):
        edict[ele] = i
    token = list()
    for i in range(len(protein)-kmer+1):
        try:
            token.append(edict[protein[i:i+kmer]])
        except KeyError:
            token.append(0)
            print("Key error : ", protein[i:i+kmer], i)
    return token


def neggen(protein, num_part=20, keep=8, max_class=4):
    length = len(protein)
    # get part
    part_len = length // num_part
    if part_len * num_part < length:
        num_part += 1

    iterator = np.arange(num_part)
    keep_parts = random.sample(list(iterator), k=keep)

    outpro = list()
    for it in iterator:
        start = it * part_len
        pro_part = protein[start:start + part_len]
        if it in keep_parts:
            outpro.extend(pro_part)
        else:
            pro_part = random.choices(np.arange(max_class), k=len(pro_part))
            outpro.extend(pro_part)
    return outpro
