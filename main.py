import numpy as np
import pandas as pd

label_map = dict(pd.read_csv("./resources/label.csv").values.tolist())
node_list = pd.read_csv("./resources/dep_node.csv").values.tolist()

label_info = {}
for node in node_list:
    label_info[node[1]] = {'label': label_map[node[1]]} if node[1] in label_map else 'unknown'


print(label_info)
