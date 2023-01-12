from __future__ import division

import csv
import pdb
import time
import os
from itertools import zip_longest

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
import logging
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from pandas import DataFrame, ExcelWriter

logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                    filename='predict.log',
                    filemode='w',  # 模式，有 w 和 a，w 就是写模式，每次都会重新写日志，覆盖之前的日志；a 是追加模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # 日志格式

def run(dataset, gpu_no, model, runs, epochs, lr, weight_decay, early_stopping,
        rpm, version, sheet: list, logger=None):
    
    torch.cuda.set_device(gpu_no)
    # print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    valacc, val_losses, accs, durations = [], [], [], []
    # epoch_time = []
    pre_list = []
    for _ in range(runs):
        data = dataset[0]
        data = data.to(device)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []
        flag = 0
        for epoch in range(1, epochs + 1):
            flag = flag + 1
            train(model, optimizer, data)
            eval_info, pre_list = evaluate(model, data)
            eval_info['epoch'] = epoch

            if logger is not None:
                logger(eval_info)

            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                val_acc = eval_info['val_acc']
                test_acc = eval_info['test_acc']
                output_model_res(data, pre_list, rpm, version, sheet)
                print('update test_acc------------------------------------------------------')
            # print(epoch)
            print(best_val_loss, test_acc)
            val_loss_history.append(eval_info['val_loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        valacc.append(val_acc)
        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)
        print('finish the epoch, duration {:.4f}------------------------------------------------------'.format(t_end - t_start))
        # print((t_end - t_start)/flag)
        # exit()

    vacc, loss, acc, duration = tensor(valacc), tensor(val_losses), tensor(accs), tensor(durations)

    print('Val Acc: {:.4f}, Val Loss: {:.4f}, Test Accuracy: {:.4f} ± {:.4f}, Duration: {:.4f}'.
          format(vacc.mean().item(),
                 loss.mean().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 duration.mean().item()))

    write_excel('/Users/zourunxin/Mine/Seminar/20Data/{}/DiGCN/main_result.xlsx'.format(version),
                '_'.join(sheet), ['Val Acc', 'Val Loss', 'Test Accuracy', 'Duration'],
                [['{:.4f}'.format(vacc.mean().item()), '{:.4f}'.format(loss.mean().item()), '{:.4f}'.format(acc.mean().item()), '{:.4f}'.format(duration.mean().item())]])
    return loss.mean().item(), acc.mean().item(), acc.std().item(), duration.mean().item()


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data)     # 喂入模型获得每个样本在每一分类下的预测值

    outs = {}
    eval_list = []
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]     # mask 即为抽样样本的索引，max(1) 按行取最大值，[1] 是每行最大值的索引。即为该样本的分类结果
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        if key == 'test':
            eval_list = pred.numpy().tolist()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

    return outs, eval_list


def output_model_res(data, predict, rpm, version, sheet: list):
    actual = data.y[data['test_mask']].numpy().tolist()
    eva = classification_report(actual, predict, output_dict=True)
    res = []
    for key, value in eva.items():
        if key == '-1':
            continue
        try:
            tmp = [key, format(value['precision'], '.3f'), format(value['recall'], '.3f'), format(value['f1-score'], '.3f'), value['support']]
            res.append(tmp)
        except Exception:
            break
    res.append(['accuracy: ' + str(eva['accuracy'])])
    res.insert(0, ['label', 'precision', 'recall', 'f1-score', '该层的包个数'])
    res.extend([[' '], [' '], [' ']])

    if len(set(actual)) > 4:
        header = ['基础环境', '核心库、核心服务', '核心工具', '系统库', '系统服务', '系统工具', '应用库', '应用服务', '应用工具', '其它']
    else:
        header = ['核心', '系统', '应用', '其它']
    matrix = confusion_matrix(actual, predict, labels=list(sorted(set(actual))))
    matrix = np.insert(matrix, len(matrix[0]), matrix.sum(axis=1), axis=1)
    tmp = matrix.tobytes()
    matrix = np.fromstring(tmp, dtype=int).reshape(len(matrix), len(matrix[0])).astype(str)
    matrix = np.insert(matrix, 0, header, axis=1)
    matrix = np.insert(matrix, 0, ['confusion_matrix'] + header + ['sum'], axis=0)
    # 将 classification_report、confusion_matrix 拼接一起写到同一个 sheet
    res.extend(matrix)
    write_excel('/Users/zourunxin/Mine/Seminar/20Data/{}/DiGCN/result_{}.xlsx'.format(version, rpm),
                '_'.join(sheet), ['']*(len(set(actual))+2), res)
    # write_excel('/Users/zourunxin/Mine/Seminar/20Data/{}/DiGCN/result_{}.xlsx'.format(version, rpm),
    #             '_'.join(sheet), ['label', 'precision', 'recall', 'f1-score', '该层的包个数'], res)
    # write_excel('/Users/zourunxin/Mine/Seminar/20Data/{}/DiGCN/confusion_matrix_{}.xlsx'.format(version, rpm),
    #             '_'.join(sheet),
    #             ['', '1', '2', '3', '4', 'sum'], matrix)


def write_excel(file: str, sheet_name: str, first_row: list, res: list):
    """
    追加模式写 xlsx，在已有的 xlsx 文件下追加 sheet 文件。若没有该 xlsx 文件，则新建并写入指定 sheet
    :param file:
    :param sheet_name:
    :param first_row:
    :param res:
    :return:
    """
    res = DataFrame(res)
    if os.path.exists(file) is False:
        with ExcelWriter(file) as writer:
            res.to_excel(writer, sheet_name, header=first_row, index=False, engine="openpyxl")
        return
    with ExcelWriter(file, mode='a', if_sheet_exists="replace") as writer:
        res.to_excel(writer, sheet_name, header=first_row, index=False, engine="openpyxl")
    return
