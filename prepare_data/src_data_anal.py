from utils.CommonUtils import convert_label, get_num_label_map
from utils.FileUtil import csv_reader, write_excel, xlrd_reader
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        # 同时显示数值和占比的饼图
        return '{p:.2f}% ({v:d})'.format(p=pct, v=val)

    return my_autopct


def pkg_statistic():
    """
    统计包的标签分布
    """
    plt.rcParams['figure.figsize'] = (12.0, 12.0)  # 设置figure_size尺寸
    j = 1
    sheets = ['含有向上层边的包', '只有向上层边的包', '含有向下层边的包', '只有向下层边的包']
    for sheet in sheets:
        reader = xlrd_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/analy_src_data/specificEdge_pkg.xlsx', sheet)
        layer_cnt = {}
        for i in range(1, reader.nrows):
            layer = reader.row_values(i)[1]
            cnt = layer_cnt.get(layer, 0) + 1
            layer_cnt[layer] = cnt
        plt.subplot(2, 2, j)
        plt.pie(layer_cnt.values(), labels=layer_cnt.keys(), autopct=make_autopct(layer_cnt.values()))
        plt.title('{}标签分布'.format(sheet))
        j += 1
    plt.tight_layout(pad=1.08)
    plt.savefig('/Users/zourunxin/Mine/Seminar/20Data/1228/analy_src_data/pkg_statistic.jpg'.format(version))


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
    plt.savefig('/Users/zourunxin/Mine/Seminar/20Data/{}/src_data_analy/label_num_statistic.jpg'.format(version))
    return


def edge_statistic():
    """
    统计各种边（同层边、相邻层边等）的数量，并画出饼状图
    统计各种包（仅有同层边的包、不含边的包等）的数量，并画出饼状图
    """
    plt.rcParams['figure.figsize'] = (12.0, 12.0)  # 设置figure_size尺寸
    reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/datasource_1228_rpm.csv')
    pkg_label = {}
    for line in reader:
        pkg_label[line[0]] = convert_label(line[1], mode='layer')

    reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/rpm/{}edges.csv'.format(legal))
    label_num = {'核心': 0, '系统': 1, '应用': 2, '其它': 3}
    layer_num = {}
    pkg_layerDiff = {}
    edges = []
    for line in reader:
        # 统计各种边的数量
        layer_diff = label_num[pkg_label[line[0]]] - label_num[pkg_label[line[1]]]
        cnt = layer_num.get(layer_diff, 0) + 1
        layer_num[layer_diff] = cnt
        # 统计各包的边的种类
        # 出度
        layerDiff = pkg_layerDiff.get(line[0], [])
        layerDiff.append(layer_diff)
        pkg_layerDiff[line[0]] = list(set(layerDiff))
        # 入度
        layerDiff = pkg_layerDiff.get(line[1], [])
        layerDiff.append(layer_diff)
        pkg_layerDiff[line[1]] = list(set(layerDiff))
        # 所有边
        edges.append([line[0], line[1]])

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

    nums = [0] * 4
    pkgs = []
    for pkg, layerDiff in pkg_layerDiff.items():
        # 仅有同层边的包
        if len(set(layerDiff)) == 1 and layerDiff.__contains__(0):
            nums[0] += 1
        # 同层-不同层边均有的包
        elif len(set(layerDiff)) > 1 and layerDiff.__contains__(0):
            nums[1] += 1
        # 仅有不同层边的包
        elif layerDiff.__contains__(0) is False:
            nums[2] += 1
            pkgs.append(pkg)
    write_excel('/Users/zourunxin/Mine/Seminar/20Data/1228/analy_src_data/specificEdge_pkg.xlsx', '只有不同层边的包', ['pkg'], pkgs)
    # 计算孤立（不含边）的包数量
    G = nx.Graph()
    G.add_nodes_from(pkg_label.keys())
    G.add_edges_from(edges)
    nums[3] = len(list(nx.isolates(G)))
    # 画包种类图
    plt.subplot(2, 2, 2)
    plt.pie(nums, labels=['仅有同层边的包数量', '同层-不同层边均有的包数量', '仅有不同层边的包数量', '没有边的包数量'], autopct=make_autopct(nums))
    plt.title('{}_num'.format('pkg_edge'))
    plt.savefig('/Users/zourunxin/Mine/Seminar/20Data/{}/analy_src_data/{}edge_num_statistic.jpg'.format(version, legal))
    return


def subGraph_statistic():
    """
    统计图的连通分量个数
    """
    reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/datasource_1228_rpm.csv')
    pkg_label = {}
    for line in reader:
        pkg_label[line[0]] = line[1]
    reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/rpm/edges.csv')
    edges = []
    for line in reader:
        edges.append([line[0], line[1]])
    G = nx.Graph()
    G.add_edges_from(edges)
    print('连通分量分别是：')
    for c in nx.connected_components(G):
        node_num = len(G.subgraph(c).nodes())
        edge_num = len(G.subgraph(c).edges())
        if node_num > 100:
            print('结点个数{}, 边数{}'.format(node_num, edge_num))
            continue
        labels = {}
        layers = {}
        classs = {}
        for n in c:
            label = pkg_label[n]
            cnt = labels.get(label, 0) + 1
            labels[label] = cnt
            cnt = layers.get(convert_label(label, mode='layer'), 0) + 1
            layers[convert_label(label, mode='layer')] = cnt
            cnt = classs.get(convert_label(label, mode='class'), 0) + 1
            classs[convert_label(label, mode='class')] = cnt
        print('结点个数{}, 边数{}, 层分布{}, 类分布{}, 标签分布{}'.format(node_num, edge_num, layers, classs, labels))
    print('共有 {} 个连通分量'.format(nx.number_connected_components(G)))
    return


def specificEdge_pkg():
    sheets = ['只有不同层边的包', '只有向上一层边的包', '只有向上两层边的包', '只有向上三层边的包', '含有向上一层边的包', '含有向上两层边的包',
              '含有向上三层边的包', '含有向上层边的包', '只有向上层边的包', '含有向下层边的包', '只有向下层边的包', '含有向下一层边的包',
              '含有向下两层边的包', '含有向下三层边的包', '只有向下一层边的包', '只有向下两层边的包', '只有向下三层边的包']
    plt.rcParams['figure.figsize'] = (12.0, 12.0)  # 设置figure_size尺寸
    reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/datasource_1228_rpm.csv')
    pkg_label = {}
    for line in reader:
        pkg_label[line[0]] = convert_label(line[1], mode='layer')

    reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/rpm/{}edges.csv'.format(legal))
    label_num = {'核心': 0, '系统': 1, '应用': 2, '其它': 3}
    pkg_layerDiff = {}
    for line in reader:
        # 统计各包的边的种类
        layer_diff = label_num[pkg_label[line[0]]] - label_num[pkg_label[line[1]]]
        # 出度
        layerDiff = pkg_layerDiff.get(line[0], [])
        layerDiff.append(layer_diff)
        pkg_layerDiff[line[0]] = list(set(layerDiff))

    pkgs = [[] for _ in range(17)]
    for pkg, layerDiff in pkg_layerDiff.items():
        # 只有向上一层边的包
        if layerDiff.__contains__(-1) and len(layerDiff) == 1:
            pkgs[1].append([pkg, pkg_label[pkg]])
        # 只有向上两层边的包
        elif layerDiff.__contains__(-2) and len(layerDiff) == 1:
            pkgs[2].append([pkg, pkg_label[pkg]])
        # 只有向上三层边的包
        elif layerDiff.__contains__(-3) and len(layerDiff) == 1:
            pkgs[3].append([pkg, pkg_label[pkg]])
        # 含有向上一层边的包
        if layerDiff.__contains__(-1):
            pkgs[4].append([pkg, pkg_label[pkg]])
        # 含有向上两层边的包
        if layerDiff.__contains__(-2):
            pkgs[5].append([pkg, pkg_label[pkg]])
        # 含有向上三层边的包
        if layerDiff.__contains__(-3):
            pkgs[6].append([pkg, pkg_label[pkg]])
        # 含有向上层边的包
        if len(set(layerDiff) & {-1, -2, -3}) > 0:
            pkgs[7].append([pkg, pkg_label[pkg]])
        # 只有向上层边的包
        if len(set(layerDiff) & {0, 1, 2, 3}) == 0 and len(layerDiff) > 0:
            pkgs[8].append([pkg, pkg_label[pkg]])
        # 含有向下层边的包
        if len(set(layerDiff) & {1, 2, 3}) > 0:
            pkgs[9].append([pkg, pkg_label[pkg]])
        # 只有向下层边的包
        if len(set(layerDiff) & {0, -1, -2, -3}) == 0 and len(layerDiff) > 0:
            pkgs[10].append([pkg, pkg_label[pkg]])
        # 含有向下一层边的包
        if layerDiff.__contains__(1):
            pkgs[11].append([pkg, pkg_label[pkg]])
        # 含有向下两层边的包
        if layerDiff.__contains__(2):
            pkgs[12].append([pkg, pkg_label[pkg]])
        # 含有向下三层边的包
        if layerDiff.__contains__(3):
            pkgs[13].append([pkg, pkg_label[pkg]])
        # 只有向下一层边的包
        if layerDiff.__contains__(1) and len(layerDiff) == 1:
            pkgs[14].append([pkg, pkg_label[pkg]])
        # 只有向下两层边的包
        if layerDiff.__contains__(2) and len(layerDiff) == 1:
            pkgs[15].append([pkg, pkg_label[pkg]])
        # 只有向下三层边的包
        if layerDiff.__contains__(3) and len(layerDiff) == 1:
            pkgs[16].append([pkg, pkg_label[pkg]])

    for sheet, pkg in zip(sheets, pkgs):
        write_excel('/Users/zourunxin/Mine/Seminar/20Data/1228/analy_src_data/specificEdge_pkg.xlsx', sheet, ['pkg', 'layer'], pkg)
    return


if __name__ == '__main__':
    version = '1228'
    legal = ''
    edge_statistic()