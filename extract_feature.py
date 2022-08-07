import csv
import os

nodes = []


with open('dep_node.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for row in csv_reader:
        nodes.append(row)

print(nodes)
for node in nodes:
    print('depname:{}'.format(node[1]))
    os.system('rpm -qi {}'.format(node[1]))

