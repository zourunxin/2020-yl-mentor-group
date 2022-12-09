import csv
import xlrd
from itertools import zip_longest
from preprocess import get_label
from pandas import DataFrame, ExcelWriter
import os


def dot_reader(file):
    with open(file, 'r') as f:
        reader = f.readlines()
    return reader


def txt_reader(file):
    f = open(file, "r", encoding='utf-8-sig')
    reader = f.readlines()
    f.close()
    return reader


def csv_reader(file):
    reader = csv.reader(open(file, encoding='utf-8-sig'))
    next(reader)
    return reader


def xlrd_reader(file):
    sheet = xlrd.open_workbook(file).sheet_by_name("Sheet1")
    return sheet


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


def csv_writer(file: str, row: list):
    """
    创建一个 writer 并返回，该 writer 已写上首行
    :param file:
    :param row:
    :return:
    """
    writer = csv.writer(open(file, 'w', encoding='utf-8-sig', newline=''))
    writer.writerow(row)
    return writer


def write_csv(file: str, first_row: list, res: list):
    writer = csv_writer(file, first_row)
    for row in res:
        writer.writerow(row)


def write_csv_with_col_first(file, first_row, res):
    """
    列写 csv
    :param res: 二维 list
    :param file: 文件地址
    :param first_row: 首行
    :return:
    """
    writer = csv_writer(file, first_row)
    for row in zip_longest(*res):  # * 用于取出列表中的每一个元素；zip_longest 取出每一列表相应位置的元素组成元组，若某列该位置不存在元素，则赋 ‘’
        writer.writerow(row)
    return


def generate_nodes_edges(dot_file='/Users/zourunxin/Mine/Seminar/20Data/all.dot'):
    """
    基于 dot 文件生成 nodes-[[idx1,pkg1], [idx2,pkg2], ...] 和 edges-[[pkg1,pkg2], [pkg1,pkg3], ...] 文件
    :param dot_file:
    :return:
    """
    reader = dot_reader(dot_file)
    pkgs_set = set()
    edges = []
    for line in reader:
        eles = line.strip().split(' -> ')
        if len(eles) == 2:
            pkg1 = eles[0].strip().replace('"', '')
            pkg2 = eles[1].strip().replace('"', '')
            pkgs_set.add(pkg1)
            pkgs_set.add(pkg2)
            edges.append([pkg1, pkg2])
        else:
            pkg = eles[0].replace('"', '')
            pkgs_set.add(pkg)

    pkgs_set = sorted(list(pkgs_set), key=str.lower)
    pkgs_list = list()
    pkg_idx = dict()
    for i, pkg in enumerate(pkgs_set):
        pkgs_list.append([i, pkg])
        pkg_idx[pkg] = i
    edges_idx = list()
    for edge in edges:
        edges_idx.append([pkg_idx[edge[0]], pkg_idx[edge[1]]])
    write_csv('/Users/zourunxin/Mine/Seminar/20Data/nodes.csv', ['idx', 'pkg'], pkgs_list)
    write_csv('/Users/zourunxin/Mine/Seminar/20Data/edges.csv', ['pkg1', 'pkg2'], edges)
    write_csv('/Users/zourunxin/Mine/Seminar/20Data/edges_idx.csv', ['idx1', 'idx2'], edges_idx)
    return


def get_nodes(file='/Users/zourunxin/Mine/Seminar/20Data/nodes.csv') -> dict:
    """
    基于 dot 生成的 nodes.csv 获取 nodes 的 map
    :param file: 形如 [[idx1,pkg1], [idx2,pkg2], ...]
    :return: <pkg1:[idx1,pkg1], pkg2:[idx2,pkg2], ...>
    """
    reader = csv_reader(file)
    nodes_map = dict()
    for line in reader:
        nodes_map[line[1]] = line
    return nodes_map


def get_edges(file='/Users/zourunxin/Mine/Seminar/20Data/edges.csv') -> list:
    """
    基于 dot 生成的 edges.csv 获取 edges 的 map
    :param file: 形如 [[pkg1,pkg2], [pkg1,pkg3], ...]
    :return: [[pkg1,pkg2], [pkg1,pkg3], ...]
    """
    reader = csv_reader(file)
    edges = list()
    for line in reader:
        edges.append(line)
    return edges


def generate_nodes_edges2(file='/Users/zourunxin/Mine/Seminar/20Data/1008/1008协商(无内核）.xlsx'):
    """
    基于 dot 和 csv 取交集生成 node.csv、edge.csv、edge_idx.csv
    :return:
    """
    sheet = xlrd_reader(file)
    all_nodes = get_nodes()
    nodes_map = dict()
    for i in range(1, sheet.nrows):
        line = sheet.row_values(i)
        pkg = line[3]
        label = get_label([line[2], line[1], line[0]])
        if pkg in all_nodes:
            nodes_map[pkg] = [pkg, line[2], label]
    pkgs_set = sorted(nodes_map.keys(), key=str.lower)
    for i, pkg in enumerate(pkgs_set):
        line = nodes_map[pkg]
        line.insert(0, str(i))
        nodes_map[pkg] = line

    all_edges = get_edges()
    edges = list()
    edges_idx = list()
    for edge in all_edges:
        if edge[0] in nodes_map and edge[1] in nodes_map:
            edges.append(edge)
            edges_idx.append([nodes_map[edge[0]][0], nodes_map[edge[1]][0]])

    nodes = sorted(nodes_map.values(), key=lambda x: str.lower(x[1]))
    write_csv('/Users/zourunxin/Mine/Seminar/20Data/1008/nodes.csv', ['idx', 'pkg', 'source_label', 'label'], nodes)
    write_csv('/Users/zourunxin/Mine/Seminar/20Data/1008/edges.csv', ['pkg1', 'pkg2'], edges)
    write_csv('/Users/zourunxin/Mine/Seminar/20Data/1008/edges_idx.csv', ['idx1', 'idx2'], edges_idx)


if __name__ == '__main__':
    generate_nodes_edges2()
