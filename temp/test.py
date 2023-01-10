import pandas as pd

df_data = pd.read_csv("../output/datasource_1228_new.csv")

indexs = [1, 3, 5, 7]

print(df_data.loc[indexs])