import xlrd
import matplotlib.pyplot as plt
import numpy as np


def diff_paras():
    """
    尝试不同参数执行 digcn_ib.py，把结果输出成折线图
    """
    lrs = [0.001, 0.005, 0.01, 0.05, 0.1]
    weight_decay = [0.0001, 0.0005, 0.001, 0.005]
    hidden = [32, 128, 512]
    dropout = [0.1, 0.3, 0.6, 0.8]
    alpha = [0.01, 0.1, 0.3, 0.6]
    parameters = [lrs, weight_decay, hidden, dropout, alpha]
    i = 0
    plt.rcParams['figure.figsize'] = (12.0, 12.0)
    for par, def_val, name in zip(parameters, [[0.0005, 32, 0.6, 0.1], [0.01, 32, 0.6, 0.1], [0.01, 0.0005, 0.6, 0.1],
                                               [0.01, 0.0005, 32, 0.1], [0.01, 0.0005, 32, 0.6]],
                                               ['lr', 'weight_decay', 'hidden', 'dropout', 'alpha']):
        plt.subplot(3, 2, i + 1)
        y_list = []
        for val in par:
            def_val.insert(i, val)
            def_val = [str(x) for x in def_val]
            reader = xlrd_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/DiGCN/main_result.xlsx', '_'.join(def_val))
            for j in range(1, reader.nrows):
                test_acc = reader.row_values(j)[2]
                y_list.append(float(test_acc))
            del def_val[i]

        # 设置 x、y 轴区间
        plt.xlim((0, par[-1]))
        plt.ylim((0, 1))
        plt.xticks(np.arange(0, par[-1] + par[0], par[-1] / 5))
        plt.yticks(np.arange(0, 1, 0.1))
        plt.plot(par, y_list, marker='o')
        plt.xlabel(name)
        plt.ylabel('test_acc')
        i += 1
    plt.tight_layout(pad=1.08)
    plt.savefig('/Users/zourunxin/Mine/Seminar/20Data/1228/DiGCN/diff_para.jpg', bbox_inches='tight')


def xlrd_reader(file, sheet_name='Sheet1'):
    sheet = xlrd.open_workbook(file).sheet_by_name(sheet_name)
    return sheet


if __name__ == '__main__':
    diff_paras()
