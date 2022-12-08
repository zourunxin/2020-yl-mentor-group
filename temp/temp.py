import pandas as pd

src_data = pd.read_csv("../data_resources/1_src_all.csv")

src_list = src_data.values.tolist()

no_content_list = []
no_content_count = 0
content_count = 0

for data in src_list:
    if isinstance(data[5], str):
        content_count += 1
    else:
        no_content_list.append(data[0])
        no_content_count += 1

print(no_content_count)
print(content_count)
print(no_content_list)