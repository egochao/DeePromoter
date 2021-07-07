import torch
import math
import torch.optim as optim
from icecream import ic
from pathlib import Path
from torch import nn

from dataloader import load_data
from modules.deepromoter import DeePromoter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(net, loaders):
    eval_result = list()
    for load in loaders:
        total = 0
        correct = 0
        for data in load:
            inputs = data[0]
            labels = data[1]
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = correct/total
        eval_result.append(acc)
    TP = eval_result[0]
    TN = 1 - eval_result[0]
    FP = eval_result[1]
    FN = 1 - eval_result[1]

    precision = TP / (TP + TN)
    recall = TP / (TP + TN)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return precision, recall, MCC


def train(data_path, pretrain=None, exp_name="test", training=True):
    epoch_num = 100000
    ker = [27, 14, 7]

    # create the experiment folder to save the result
    output = Path("./output")
    output.mkdir(exist_ok=True)
    exp_folder = output.joinpath(exp_name)
    exp_folder.mkdir(exist_ok=True)

    # load data
    data = load_data(data_path, device=device)
    train_pos, val_pos, test_pos, train_neg, val_neg, test_neg = data

    # model define
    net = DeePromoter(ker)
    net.to(device)

    # load pre-train model
    if pretrain is not None:
        net.load_state_dict(torch.load(pretrain))

    # define loss, optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.00001)

    running_loss = 0
    # pbar = tqdm(range(epoch_num))
    pbar = range(epoch_num)
    ic("start loop")
    if training:
        for epoch in pbar:
            for i, (batch_pos, batch_neg) in enumerate(zip(train_pos, train_neg)):
                inputs = torch.cat((batch_pos[0], batch_neg[0]), dim=0)
                labels = torch.cat((batch_pos[1], batch_neg[1]), dim=0)

                # zero the parameter gradients
                optimizer.zero_grad()

                # pass model to
                outputs = net(inputs)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            if epoch % 10 == 0:
                torch.save(net.state_dict(), str(exp_folder.joinpath("epoch_" + str(epoch) + ".pth")))
                net.eval()
                precision, recall, MCC = test(net, [val_pos, val_neg])
                net.train()
                ic("Epoch :", epoch)
                ic("precision :", precision)
                ic("recall :", recall)
                ic("MCC :", MCC)


    # test
    net.eval()
    precision, recall, MCC = test(net, [test_pos, test_neg])
    ic("precision :", precision)
    ic("recall :", recall)
    ic("MCC :", MCC)


if __name__ == '__main__':
    path = "./data/human/nonTATA/hs_pos_nonTATA.txt"
    # train from scratch
    train(path, pretrain=pretrain, exp_name="test", training=True)

    # test
    train(path, pretrain=pretrain, exp_name="test", training=True)


