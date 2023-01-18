from utils.CommonUtils import convert_label, get_num_label_map
from utils.FileUtil import csv_reader, write_excel
import matplotlib.pyplot as plt
import numpy as np


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        # 同时显示数值和占比的饼图
        return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)

    return my_autopct


def label_statistic():
    """
    统计各 label 下包的数量，并画出饼状图
    """
    plt.rcParams['figure.figsize'] = (12.0, 12.0)  # 设置figure_size尺寸
    i = 1
    for mode, keys in zip(['layer', 'class', 'both'],
                          [['核心', '系统', '应用', '其它'], ['库', '工具', '服务', '其它'],
                           ["基础环境", "核心库", "核心工具", "核心服务", "系统服务", "系统库", "系统工具", "应用服务", "应用库", "应用工具", '其它']]):
        plt.subplot(2, 2, i)
        reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/datasource_1228_rpm.csv')
        layer_cnt = {key: 0 for key in keys}
        for line in reader:
            layer = convert_label(line[1], mode=mode)
            layer_cnt[layer] += 1

        layer_num = np.array([x for x in layer_cnt.values()])
        plt.pie(layer_num, labels=keys, autopct=make_autopct(layer_num))
        plt.title('{}_num'.format(mode))
        i += 1
    plt.tight_layout(pad=1.08)
    plt.savefig('/Users/zourunxin/Mine/Seminar/20Data/{}/src_data_analy/label_num_statistic.jpg'.format(version, mode))
    return


def edge_statistic():
    """
    统计各种边的数量，并画出饼状图
    """
    plt.rcParams['figure.figsize'] = (12.0, 12.0)  # 设置figure_size尺寸
    reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/datasource_1228_rpm.csv')
    pkg_label = {}
    for line in reader:
        pkg_label[line[0]] = convert_label(line[1], mode='layer')

    reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/rpm/edges.csv')
    label_num = {'核心': 0, '系统': 1, '应用': 2, '其它': 3}
    layer_num = {}
    pkg_layerDiff = {}
    for line in reader:
        # 统计各种边的数量
        layer_diff = label_num[pkg_label[line[0]]] - label_num[pkg_label[line[1]]]
        cnt = layer_num.get(layer_diff, 0) + 1
        layer_num[layer_diff] = cnt
        # 统计各包的边的种类
        layerDiff = pkg_layerDiff.get(line[0], [])
        layerDiff.append(layer_diff)
        pkg_layerDiff[line[0]] = list(set(layerDiff))

    # 构造驼形数组，使饼状图数字不被紧挨在一起
    layer_num = sorted(layer_num.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    i = 0; j = len(layer_num) - 1
    nums = []; labels = []
    while i < j:
        nums.append(layer_num[i][1])
        nums.append(layer_num[j][1])
        labels.append(layer_num[i][0])
        labels.append(layer_num[j][0])
        i += 1; j -= 1
    if i == j:
        nums.append(layer_num[i][1])
        labels.append(layer_num[i][0])
    # 画边数量图
    plt.subplot(2, 2, 1)
    plt.pie(nums, labels=labels, autopct=make_autopct(nums))
    plt.title('{}_num'.format('edge'))

    nums = [0] * 3; labels = ['仅有同层边的包数量', '同层-不同层边均有的包数量', '仅有不同层边的包数量', '没有边的包数量']
    pkg0 = []; pkg1 = []; pkg2 = []
    for pkg, layerDiff in pkg_layerDiff.items():
        if len(set(layerDiff)) == 1 and layerDiff.__contains__(0):
            nums[0] += 1
            pkg0.append(pkg)
        elif len(set(layerDiff)) > 1 and layerDiff.__contains__(0):
            nums[1] += 1
            pkg1.append(pkg)
        elif layerDiff.__contains__(0) is False:
            nums[2] += 1
            pkg2.append(pkg)
        else:
            print(pkg)
        del pkg_label[pkg]   # 剩下未被删除的包即为不含边的包
    # 打印三种边对应的包
    for res, sheet in zip([pkg0, pkg1, pkg2, pkg_label.keys()], labels):
        write_excel('/Users/zourunxin/Mine/Seminar/20Data/1228/analy_src_data/pkgs.xlsx', sheet, ['pkg'], res)
    # 画包种类图
    plt.subplot(2, 2, 2)
    nums.append(len(pkg_label))
    plt.pie(nums, labels=labels, autopct=make_autopct(nums))
    plt.title('{}_num'.format('pkg_edge'))
    plt.savefig('/Users/zourunxin/Mine/Seminar/20Data/{}/analy_src_data/edge_num_statistic.jpg'.format(version))
    return


if __name__ == '__main__':
    version = '1228'
    edge_statistic()