import sys
sys.path.append("../")
import os
import utils.NLPUtils as NLPUtils
import utils.CommonUtils as CommonUtils
import pandas as pd
import argparse

def trans2str(data):
    text = str(data)
    return text if text != "nan" else ""

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default = "rpm") # rpm: 二进制包, src: 源码包
parser.add_argument('--type', type=str, default = "both") # class: 分类, layer: 分层, both: 分类分层
args = parser.parse_args()


print("正在预处理数据，数据源：{}，任务：{}".format(args.source, args.type))

src_dir = "../data_resources/1228_src.csv" if args.source == "src" else "../data_resources/1228_bin.csv"
df = pd.read_csv(src_dir)
# 预处理文本
df["name"] = df["src_name" if args.source == "src" else "rpm_name"]
df["summary"] = df.apply(lambda x: trans2str(x["zero_summary"]) + " " + trans2str(x["summary"]), axis=1)
df["description"] = df.apply(lambda x: trans2str(x["zero_description"]) + " " + trans2str(x["description"]), axis=1)
df["text"] = df.apply(lambda x: x["summary"] + " " + x["description"], axis=1)
df["text"] = df.text.apply(NLPUtils.remove_seperator)

# 预处理标签
df["label"] = df["分层分类"].apply(lambda x: CommonUtils.convert_label(x.split("/")[0].strip(), mode=args.type))

df_output = df[["name", "label", "text", "summary", "description"]]
df_output.to_csv('../output/datasource_1228.csv')
print("数据预处理完毕，软件包个数：{}".format(len(df_output)))

# 依赖关系预处理
def resolve_dep(dot_dir):
    with open(dot_dir, 'r') as f:
        lines = f.readlines()
    deps = []
    for line in lines:
        _line = line.strip().split(' -> ')
        if (len(_line) == 2):
            deps.append(line.strip())
    return deps

def write_dot(pkgs, deps):
    f = open('../output/all.dot', 'w')
    f.write('digraph MyPicture {\n')
    f.write('\tgraph [rankdir=LR]\n')
    for name in pkgs:
        f.write('\t' + name + '\n')
        for dep in deps:
            if (name == dep.split(' -> ')[0].replace('"', "")):
                f.write('\t' + dep + '\n')
    f.write('}')
    f.close()


print("正在预处理依赖关系")
g = os.walk(r"../data_resources/dot")
dot_dirs = []
for path, dir_list, file_list in g:
    for file_name in file_list:
        dot_dirs.append(os.path.join(path, file_name))

dots_deps = []
for dir in dot_dirs:
    dots_deps += resolve_dep(dir)
dots_deps = list(set(dots_deps))
print("dot 文件中读取依赖 {} 个".format(len(dots_deps)))
# dot 文件里的所有出现的包的集合
pkg_names = list(set([dep.split(' -> ')[0].replace('"', '') for dep in dots_deps]))
pkg_names += list(set([dep.split(' -> ')[1].replace('"', '')for dep in dots_deps]))

write_dot(pkg_names, dots_deps)
filtered_deps = []
for dep in dots_deps:
    dep = dep.split(' -> ')
    _dep = [dep[0].replace('"', ''), dep[1].replace('"', '')]
    name_set = set(df_output["name"])
    if _dep[0] in name_set and _dep[1] in name_set:
        filtered_deps.append(_dep)
    else:
        print("依赖关系: {} -> {} 不在数据中".format(_dep[0], _dep[1]))


df_deps = pd.DataFrame(filtered_deps,columns=['out','in'])
df_deps.to_csv('../output/edges.csv')
print("依赖关系处理完成，依赖数：{}".format(len(filtered_deps)))