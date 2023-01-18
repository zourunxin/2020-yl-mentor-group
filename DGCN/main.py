# 执行不同参数下 digcn.py 的运行结果，并画出折线图
import os
import time


def diff_train_num():
    """
    不同训练集大小下模型结果的规律
    """
    start = time.time()
    train_nums = [20, 50, 100, 150, 200]
    for par in train_nums:
        sheet = '{}train_num'.format(par)
        sent = 'python digcn_ib.py --dataset rpm --npz 1228layerditfidf2000feature --train_num {} --sheet {}'.format(par, sheet)
        print('\n' + '\n' + '\n' + '执行 {} 语句'.format(sent))
        os.system(sent)
        print('花费时间 {}min'.format(str((time.time() - start) / 60)))
    print('执行时间 ' + str((time.time() - start) / 60) + 'min')
    return


if __name__ == '__main__':
    diff_train_num()