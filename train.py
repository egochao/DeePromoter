import torch
import math
import argparse
import torch.optim as optim
from torch import nn
from icecream import ic
from pathlib import Path

from dataloader import load_data
from modules.deepromoter import DeePromoter
from test import evaluate, mcc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(data_path, pretrain=None, exp_name="test", training=True, ker=None, epoch_num=1000):
    """
    Training
    :param data_path: Path to the txt data file
    :param pretrain: Path to weight for continue training
    :param exp_name: Folder name to save the results
    :param training: If False, performs testing only
    :param ker: List kernel size of list CNN applying to the protein sequence
    :param epoch_num: Max epoch to train
    """
    if ker is None:
        ker = [27, 14, 7]

    # create the experiment folder to save the result
    output = Path("./output")
    output.mkdir(exist_ok=True)
    exp_folder = output.joinpath(exp_name)
    exp_folder.mkdir(exist_ok=True)

    # load data
    ic("Data loading")
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
    best_mcc = 0
    best_precision = 0
    best_recall = 0
    break_after = 10
    last_update_best = 0
    pbar = range(epoch_num)
    ic("Start training")
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

                running_loss += loss.item()

            if epoch % 10 == 0:
                torch.save(net.state_dict(), str(exp_folder.joinpath("epoch_" + str(epoch) + ".pth")))
                net.eval()
                eval_data, _ = evaluate(net, [val_pos, val_neg])
                precision, recall, MCC = mcc(eval_data)
                net.train()
                ic("Epoch :", epoch, "Experiment :", exp_name)
                ic("precision :", precision)
                ic("recall :", recall)
                ic("MCC :", MCC)

                # save best model
                if precision > best_precision:
                    best_precision = precision
                    ic("Update best precision")
                    torch.save(net.state_dict(), str(exp_folder.joinpath("best_precision.pth")))
                if recall > best_recall:
                    best_recall = recall
                    ic("Update best recall")
                    torch.save(net.state_dict(), str(exp_folder.joinpath("best_recall.pth")))
                if MCC > best_mcc:
                    ic("Update best MCC")
                    best_mcc = MCC
                    torch.save(net.state_dict(), str(exp_folder.joinpath("best_mcc.pth")))
                    last_update_best = 0
                else:
                    last_update_best += 1
                if last_update_best >= break_after:
                    break
            if last_update_best >= break_after:
                break

    # test
    best_model = str(exp_folder.joinpath("best_mcc.pth"))
    net.load_state_dict(torch.load(best_model))
    net.eval()
    eval_data, _ = evaluate(net, [test_pos, test_neg])
    precision, recall, MCC = mcc(eval_data)
    ic("precision :", precision)
    ic("recall :", recall)
    ic("MCC :", MCC)
    with open("log.txt", "w") as f:
        f.write(f"Test precision: {precision}\n")
        f.write(f"Test recall: {recall}\n")
        f.write(f"Test MCC : {MCC}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str,
                        required=True, help="path to dataset(txt file)")
    parser.add_argument("-e", "--experiment_name", type=str,
                        default="test", help="name of folder to save output in ./output")
    parser.add_argument("-w", "--weight", type=str, help="Path to pre-train")
    parser.add_argument("--test", action="store_true", help="Add this flag to do test only")
    args = parser.parse_args()

    training = True
    if args.test:
        training = False
    train(args.data, pretrain=args.weight, exp_name=args.experiment_name, training=training)
