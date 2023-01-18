import pandas as pd

from utils import CommonUtils

df_data = pd.read_csv('../output/datasource_1228_new.csv')
df_edges = pd.read_csv('../output/edges.csv')
idx_name_map, name_idx_map = CommonUtils.get_idx_name_map(df_data["name"])
out = df_edges['out'].apply(lambda x: name_idx_map[x])
ind = df_edges['in'].apply(lambda x: name_idx_map[x])
print(out)
print(ind)