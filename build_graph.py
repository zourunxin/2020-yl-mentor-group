import os
import re
import csv

f1=open('dependencies.txt', encoding='gbk')
dependencies=[]
for line in f1:
    dependencies.append(line.strip())


f2=open('pacts.txt', encoding='gbk')
pacts=[]
for line in f2:
    pacts.append(line.strip())

pacts = [line.split()[0].split('.')[0] for line in pacts]

dependencies = list(filter(lambda d: not d.find('error') == 0 , dependencies))

deps = []
d_name = ''
reg = '-[0-9]'

for d in dependencies:
    if (d.find('depname') == 0):
        d_name = d.split(':')[1]
    else:
        dep_name = d.split()[-1]
        final_name = dep_name[0 : re.search(reg, dep_name).span()[0]]
        deps.append('{} -> {}'.format(final_name, d_name))

node = pacts
edge = list(set(deps))
edge = [e.split(' -> ') for e in edge]

map = {}

for i in range(len(node)):
    map[node[i]] = i
    node[i] = [i, node[i]]

edge = [[map.get(e[0]), map.get(e[1])] for e in edge]

edge.sort()


with open("dep_node.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["idx", "name"])   # 先写入columns_name
    writer.writerows(node)   #写入多行用writerows

with open("dep_edge.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["idx1", "idx2"])   # 先写入columns_name
    writer.writerows(edge)   #写入多行用writerows



