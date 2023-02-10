from utils.FileUtil import csv_reader, write_csv, dot_reader
from utils.ModelEvaluationUtil import main2
from utils.util import get_label_layer_map


def pkg_vector():
    # 统计各个包描述的单词向量
    pkgs = {}
    reader = csv_reader(file)
    for line in reader:
        pkg = line[0]
        pkgs[pkg] = statistic_word(line[2])
    return pkgs


def label_vector_finely():
    """
    统计 11 分类下各标签描述的单词向量
    :return:
    """
    # 统计各个标签的所有描述
    labels = {}
    reader = csv_reader(file)
    for line in reader:
        label = line[1]
        text = line[2].replace('\\\n', '')
        label_text = labels.get(label, "") + text
        labels[label] = label_text

    # 统计各个标签描述的单词的个数
    for label, text in labels.items():
        labels[label] = statistic_word(text)

    for word_cnt in labels.values():
        sorted(word_cnt.items(), key=lambda kv: (kv[1], kv[0]))
    return labels


def layer_vector():
    """
    统计 5 分类下各标签描述的单词向量
    :return:
    """
    # 统计各个标签的所有描述
    layers = {}
    reader = csv_reader(file)
    label_layer_map = get_label_layer_map()
    for line in reader:
        layer = label_layer_map.get(line[1], 4)
        text = line[2].replace('\\\n', '')
        layer_text = layers.get(layer, "") + text
        layers[layer] = layer_text

    # 统计各个层描述的单词的个数
    for layer, text in layers.items():
        layers[layer] = statistic_word(text)

    for word_cnt in layers.values():
        sorted(word_cnt.items(), key=lambda kv: (kv[1], kv[0]))
    return layers


def statistic_word(text: str):
    tmp = {}
    for word in text.strip().split():
        cnt = tmp.get(word, 0) + 1
        tmp[word] = cnt
    return tmp


def most_match(pkgs, labels):
    """
    pkg: [14, 12, 9, ...]   label: [1, 0, 0, 1, ...]
    :param pkgs: {pkg1: {word1: cnt1, word2: cnt2, ...}}
    :param labels: {label1, {word1: cnt1, ...}},
    :return:
    """
    # 找出包的最近似类别
    res = {}
    for pkg, word_cnt1 in pkgs.items():
        max_match_cnt = 0
        for label, word_cnt2 in labels.items():
            match_cnt = 0
            for word, cnt in word_cnt1.items():
                if word in word_cnt2:
                    match_cnt += cnt
            if match_cnt > max_match_cnt:
                res[pkg] = label
                max_match_cnt = match_cnt
    return res


def pkg_label(file: str):
    res = {}
    reader = csv_reader(file)
    for line in reader:
        res[line[0]] = line[1]
    return res


def pkg_layer(file):
    res = {}
    reader = csv_reader(file)
    label_layer_map = get_label_layer_map()
    for line in reader:
        res[line[0]] = label_layer_map.get(line[1], 4)
    return res


def statistic_edges():
    """
    统计各 label 的边的个数
    :return:
    """
    pkg_label_map = pkg_label('/Users/zourunxin/Mine/Seminar/20Data/1128/output/datasource_1128.csv')
    label_edges = {}   # {label1: [总边的个数, 出度个数, 入度个数], ...}
    edge_file = '/Users/zourunxin/Mine/Seminar/20Data/1128/output/all.dot'
    reader = dot_reader(edge_file)
    for line in reader:
        line = line.split(' -> ')
        if len(line) > 1:
            label1 = pkg_label_map[line[0].strip()]
            label2 = pkg_label_map[line[1].strip()]
            edges1 = label_edges.get(label1, [0]*3)
            edges1[0] += 1
            edges1[1] += 1
            label_edges[label1] = edges1
            edges2 = label_edges.get(label2, [0]*3)
            edges2[0] += 1
            edges2[2] += 1
            label_edges[label2] = edges2
    ans = []
    for label, edges in label_edges.items():
        tmp = [label]
        tmp.extend(edges)
        ans.append(tmp)
    write_csv('/Users/zourunxin/Mine/Seminar/20Data/1128/词频统计/edges_statistic.csv', ['label', '总边的个数', '出度个数', '入度个数'], ans)
    return


def layer_statistic_edges():
    """
    统计各层的边数量
    """
    pkg_layer_map = pkg_layer('/Users/zourunxin/Mine/Seminar/20Data/1128/output/datasource_1128.csv')
    layer_edges = {}
    edge_file = '/Users/zourunxin/Mine/Seminar/20Data/1128/output/all.dot'
    reader = dot_reader(edge_file)
    for line in reader:
        line = line.split(' -> ')
        if len(line) > 1:
            layer1 = pkg_layer_map[line[0].strip()]
            layer2 = pkg_layer_map[line[1].strip()]
            edges1 = layer_edges.get(layer1, [0]*3)
            edges1[0] += 1
            edges1[1] += 1
            layer_edges[layer1] = edges1
            edges2 = layer_edges.get(layer2, [0]*3)
            edges2[0] += 1
            edges2[2] += 1
            layer_edges[layer2] = edges2
    ans = []
    for label, edges in layer_edges.items():
        tmp = [label]
        tmp.extend(edges)
        ans.append(tmp)
    write_csv('/Users/zourunxin/Mine/Seminar/20Data/1128/词频统计/edges_statistic_layer.csv', ['label', '总边的个数', '出度个数', '入度个数'], ans)
    return


if __name__ == '__main__':
    layer_statistic_edges()

    file = '/Users/zourunxin/Mine/Seminar/20Data/1128/output/datasource_1128.csv'
    pkgs = pkg_vector()

    # 统计分类下的单词向量
    labels = layer_vector()

    # 找出包的最近似类别
    predict_map = most_match(pkgs, labels)
    actual_map = pkg_layer(file)
    res = main2(predict_map, actual_map)
    write_csv('/Users/zourunxin/Mine/Seminar/20Data/1128/词频统计/result.csv', ['label', '该层的包个数', '分到该层的包个数', 'precision', 'recall', 'f1-score'], res)
