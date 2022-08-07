import csv

# 用去重过的 csv 文件！！

edge = []
node = []

with open('dep_edge.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for row in csv_reader:
        edge.append(row)

with open('dep_node.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for row in csv_reader:
        node.append(row)

node_map = {}

for n in node:
    node_map[n[0]] = '_'.join(n[1].split('-')).replace('++', 'pp')

with open('dep_graph.dot', 'w', encoding='gbk') as f:
    f.write('digraph graphname {\n')
    for e in edge:
        f.write('\t{} -> {}\n'.format(node_map.get(e[0]), node_map.get(e[1])))
    f.write('}')
