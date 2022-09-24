import csv
import re

from utils.FileUtil import txt_reader, csv_reader, write_csv
from utils.utils import class_cnt


def get_pkg_idx(file):
    """
    基于 node.csv 获取 <pkg, idx> 的映射
    :param file:
    :return:
    """
    reader = csv_reader(file)
    pkg_idx = dict()
    for line in reader:
        pkg_idx[line[1]] = line[0]
    return pkg_idx


def get_label_digit_map(label_list):
    """
    获取 label - 数字 之间的映射
    :return: <label: str, idx, str>
    """

    label_dict = dict()
    for i, enum in enumerate(label_list):
        label_dict[enum] = str(i)
    return label_dict


def get_digit_label_map(label_list):
    """
    获取 数字 - label 之间的映射
    :return: <idx: str, label: str>
    """

    label_dict = dict()
    for i, enum in enumerate(label_list):
        label_dict[str(i)] = enum
    return label_dict



def keep_chinese_character_digit(s: str):
    """
    对输入 string 进行清洗，仅保留中文、字母和数字，其它字符丢弃
    :param s:
    :return:
    """
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9^_^+^.^-]")  # 匹配不是中文、大小写、数字的其他字符
    return cop.sub('', s)


def gen_feature(
        in_file="../output/name_label_feature.csv",
        pkg_idx_file="../output/nodes.csv",
        out_file="../output/feature.csv"):
    """
    清洗 feature.txt，使其 label 和 pkg 均删掉。并基于 pkg_idx 使每行 feature 放在 idx 位置
    :param in_file: row<pkg, label, keyword1, keyword2, keyword3, ...,>
    :param pkg_idx_file: row<idx, pkg>
    :param out_file: row<keyword1, keyword2, keyword3, ...,>
    :return:
    """
    reader = csv_reader(in_file, header=False)
    pkg_idx = get_pkg_idx(pkg_idx_file)
    res = [[]] * len(pkg_idx)
    for line in reader:
        pkg = line[0]
        if pkg in pkg_idx:
            res[int(pkg_idx[pkg])] = line[2:]

    write_csv(out_file, [], res)
    return


def gen_idx_label(
        src_pkg_list,
        pkg_idx_file="../output/nodes.csv",
        out_file="../output/idx_label.csv"):
    """
    获取 <pkg_idx, label> 映射关系的 label.csv
    :param in_file: row<label, pkg, summary, description>
    :param pkg_idx_file: row<pkg, pkg_idx>
    :param out_file: row<pkg_idx, label>
    :return:
    """
    pkg_idx_map = get_pkg_idx(pkg_idx_file)

    label_digit_map = {'库':0, '工具':1, '服务':2, '其它': 3}

    # label_digit_map = get_label_digit_map(label_list, label_map)
    res = list()
    tmp = set()
    for line in src_pkg_list:
        pkg = line[0]
        if pkg not in pkg_idx_map:
            continue
        pkg_idx = pkg_idx_map[pkg]
        label = keep_chinese_character_digit(line[1])
        if label.find('语言') != -1:
            label = '编程语言'
        if label in label_digit_map:
            label_idx = label_digit_map[label]
        else:
            label_idx = label_digit_map['其它']
        row = [pkg_idx, label_idx]
        if row[0] in tmp:
            print("有重复数据, id: {}".format(row[0]))
        else:
            tmp.add(row[0])
            res.append(row)
    # assert len(res) == len(pkg_idx_map)
    # for line in res:
    #     if line[0] in tmp:
    #         print(line[0])
    #     else:
    #         tmp.add(line[0])
    write_csv(out_file, ['pkg_idx', 'label_idx'], res)


def util3(file='../output/idx_label.csv'):
    """
    统计文件各标签的数量并降序打印
    :param file: row<pkg_idx, label>
    :return:
    """
    label_list = ['库', '工具', '服务', '其它']

    digit_label_map = get_digit_label_map(label_list)
    reader = csv_reader(file)
    labels = list()
    for line in reader:
        label = digit_label_map[line[1]]
        labels.append(label)
    res = class_cnt(labels)
    res['总数据量'] = sum(res.values())
    print(sorted(res.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))


def util4(
        in_file="/Users/zourunxin/Mine/Seminar/20Data/0915/feature.txt",
        pkg_idx_file="/Users/zourunxin/Mine/Seminar/20Data/0915/node.csv"):
    pkg_idx_map = get_pkg_idx(pkg_idx_file)
    pkg_set = set(pkg_idx_map.keys())
    reader = txt_reader(in_file)
    for line in reader:
        elements = line.split(' ')
        pkg = elements[0]
        if pkg.find('aajohan-comfortaa-fonts') != -1:
            print(pkg)
        try:
            pkg_set.remove(pkg)
        except KeyError:
            pass
    print(pkg_set)
