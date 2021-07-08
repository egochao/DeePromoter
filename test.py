import torch
import math
import argparse
from icecream import ic

from modules.deepromoter import DeePromoter
from dataloader import load_data_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(net, loaders):
    """
    Infer and check results against labels
    :param net: Model object in eval state
    :param loaders: List of torch dataloader for infer
    :return: List of [correct, total] for every dataloader, list of predicted results in int type
    """
    eval_result = list()
    ltotal = list()
    lcorrect = list()
    pred_result = list()
    for load in loaders:
        total = 0
        correct = 0
        pred_list = list()
        for data in load:
            inputs = data[0]
            labels = data[1]
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred_list += list(predicted.cpu().numpy())
        acc = correct/total
        eval_result.append(acc)
        lcorrect.append(correct)
        ltotal.append(total)
        pred_result.append(pred_list)
    return (lcorrect, ltotal), pred_result


def mcc(data):
    """
    Calculate Matthew correlation coeficient
    :param data: List output of evaluate with the first item is positive result and second item is negative result
    :return: Precision, recall, MCC
    """
    pos_count = data[0][0]
    neg_count = data[0][1]

    tol_pos_count = data[1][0]
    tol_neg_count = data[1][1]

    TP = pos_count
    FN = tol_pos_count - pos_count
    TN = neg_count
    FP = tol_neg_count - neg_count

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return precision, recall, MCC


def test(data_path, pretrain, ker=None):
    if ker is None:
        ker = [27, 14, 7]

    dataloader = load_data_test(data_path, device=device)

    # model define
    net = DeePromoter(ker)
    net.to(device)

    net.load_state_dict(torch.load(pretrain))

    net.eval()
    eval_data, results = evaluate(net, [dataloader])

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="path to dataset(txt file)",
    )
    parser.add_argument("-w", "--weight", type=str, help="Path to pre-train")
    args = parser.parse_args()

    output = test(args.data, args.weight)

    with open("infer_results.txt", "w") as f:
        ic("Save the results to infer_results.txt")
        for out in output[0]:
            f.write(str(out) + "\n")
