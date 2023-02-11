import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import NLPUtils
from utils.FileUtil import csv_reader, write_csv, write_excel, xlrd_reader
from utils.CommonUtils import convert_label


def del_illegal_edge():
    reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/datasource_1228_rpm.csv')
    pkg_layer = {}
    layer_digit = {'核心': 0, '系统': 1, '应用': 2, '其它': 3}
    for line in reader:
        pkg_layer[line[0]] = layer_digit[convert_label(line[1], mode='layer')]

    edges = []
    reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/rpm/edges.csv')
    for line in reader:
        if pkg_layer[line[0]] < pkg_layer[line[1]]:
            continue
        edges.append([line[0], line[1]])

    write_csv('/Users/zourunxin/Mine/Seminar/20Data/1228/rpm/legal_edges.csv', ['out', 'in'], edges)
    return


def delete_isolate_node():
    """
    删除孤立节点
    """
    reader = xlrd_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/analy_src_data/specificEdge_pkg.xlsx', '没有边的包')
    isolate_node = []
    for i in range(1, reader.nrows):
        isolate_node.append(reader.row_values(i)[0])

    reader = csv_reader('../output/datasource_1228.csv')
    res = []
    for line in reader:
        if line[0] in isolate_node:
            continue
        res.append(line)
    write_csv('../output/datasource_1228_without_isolate_node.csv', ['name', 'label', 'text', 'summary', 'description'], res)



if __name__ == '__main__':
    print(delete_isolate_node())