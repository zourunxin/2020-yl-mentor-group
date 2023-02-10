import sys
sys.path.append("../")
import os
import csv
import pandas as pd
from preprocess import gen_feature, gen_idx_label, util3
from random import shuffle
from utils.util import get_label_list

def merge_label(src_pkg_list, merge_mode="class", label_map = None):

    layer_map = {
        "基础环境": "核心",
        "核心库": "核心",
        "核心工具": "核心",
        "系统服务": "系统",
        "系统库": "系统",
        "系统工具": "系统",
        "应用服务": "应用",
        "应用库": "应用",
        "应用工具": "应用",
        "虚拟化": "应用",
        "内核": "核心"
    }

    class_map = {
        "基础环境": "服务",
        "核心库": "库",
        "核心工具": "工具",
        "系统服务": "服务",
        "系统库": "库",
        "系统工具": "工具",
        "应用服务": "服务",
        "应用库": "库",
        "应用工具": "工具",
    }

    label_list = []
    _label_map = {
        "基础环境": "基础环境",
        "核心库": "核心库",
        "核心工具": "核心工具",
        "系统服务": "系统服务",
        "系统库": "系统库",
        "系统工具": "系统工具",
        "应用服务": "应用服务",
        "应用库": "应用库",
        "应用工具": "应用工具",
    }
    if merge_mode == "class":
        _label_map = class_map
    if merge_mode == "layer":
        _label_map = layer_map
    if label_map is not None:
        _label_map = label_map

    def _convert_label(x):
        if x[1] in _label_map:
            x[1] = _label_map[x[1]]
        else:
           x[1] = "其它"
        if x[1] not in label_list:
            label_list.append(x[1])
        return x
    
    merged_list = []
    for pkg in src_pkg_list:
        merged_list.append(_convert_label(pkg))

    with open("../output/label_list.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(label_list)  # 先写入columns_name
    
    return merged_list


def sample_pkg(src_pkg_list, sample_size = 99999):
    sample_lists = {
        "库": [],
        "工具": [],
        "服务": [],
        "其它": [],
    }
    for pkg in src_pkg_list:
        sample_lists[pkg[1]].append(pkg);
    
    sampled_list = []
    for label in sample_lists:
        shuffle(sample_lists[label])
        sampled_list += sample_lists[label][0:sample_size]
    
    return sampled_list


def resolve_dep(dot_dir):
    with open(dot_dir, 'r') as f:
        lines = f.readlines()
    deps = []
    for line in lines:
        _line = line.strip().split(' -> ')
        if (len(_line) == 2):
            deps.append(line.strip())
    return deps


def merge_deps(deps, src_pkg_list):
    dots_deps = list(set(deps))

    # data 文件里的所有出现的包的集合
    pkg_name_list = []
    for pkg in src_pkg_list:
        pkg_name_list.append(pkg[0])

    # dot 文件里的所有出现的包的集合
    pkg_names = list(set([dep.split(' -> ')[0].replace('"', '') for dep in dots_deps]))
    pkg_names += list(set([dep.split(' -> ')[1].replace('"', '')for dep in dots_deps]))


    # 取交集
    pkg_names = list(set(pkg_names).intersection(set(pkg_name_list)))
    pkg_names.sort()

    # 取两端都在交集中的所有边
    dots_deps_after_filter = []
    for dep in dots_deps:
        pkg1 = dep.split(' -> ')[0].replace('"', '')
        pkg2 = dep.split(' -> ')[1].replace('"', '')
        if pkg1 in pkg_name_list and pkg2 in pkg_name_list:
            dots_deps_after_filter.append(dep)
        else:
            print(dep)

    dots_deps_after_filter = list(set(dots_deps_after_filter))

    return pkg_name_list, dots_deps_after_filter


def write_dot(pkgs, deps):
    f = open('../output/all.dot', 'w')
    f.write('digraph MyPicture {\n')
    f.write('\tgraph [rankdir=LR]\n')
    for name in pkgs:
        f.write('\t' + name + '\n')
        for dep in deps:
            if (name == dep.split(' -> ')[0]):
                f.write('\t' + dep + '\n')
    f.write('}')
    f.close()


g = os.walk(r"../data_resources/dot")

dot_dirs = []
for path, dir_list, file_list in g:
    for file_name in file_list:
        dot_dirs.append(os.path.join(path, file_name))


label_mode = "layer"

dots_deps = []
for dir in dot_dirs:
    dots_deps += resolve_dep(dir)

print(len(dots_deps))

src_pkg_list = pd.read_csv("../output/name_label_feature.csv", header=None).values.tolist()
print("读取数据样本 {} 个".format(len(src_pkg_list)))
src_pkg_list = merge_label(src_pkg_list, label_mode)
# src_pkg_list = sample_pkg(src_pkg_list)
# print("抽样数据样本 {} 个".format(len(src_pkg_list)))
pkg_names, deps = merge_deps(dots_deps, src_pkg_list)
print(len(deps))

write_dot(pkg_names, deps)


idx_pkg = []
pkg_idx_map = {}
idx1_idx2 = []

for idx, name in enumerate(pkg_names):
    idx_pkg.append([idx, name])
    pkg_idx_map[name] = idx

for dep in deps:
    dep = dep.split(' -> ')
    idx1_idx2.append([pkg_idx_map[dep[0].replace('"', '')], pkg_idx_map[dep[1].replace('"', '')]])

idx1_idx2.sort(key=lambda x: [x[0], x[1]])

print("结点数量: {}".format(len(idx_pkg)))
print("边数量: {}".format(len(idx1_idx2)))

with open("../output/nodes.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["idx", "name"])  # 先写入columns_name
    writer.writerows(idx_pkg)  # 写入多行用writerows

with open("../output/edges.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["idx1", "idx2"])  # 先写入columns_name
    writer.writerows(idx1_idx2)  # 写入多行用writerows


gen_feature()
gen_idx_label(src_pkg_list)
util3()