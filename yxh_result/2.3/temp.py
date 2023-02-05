import pandas as pd

df_old = pd.read_csv("../../output/datasource_0117_class.csv")
df_new = pd.read_csv("标签调整-20230204.csv")

old_list = df_old.values.tolist()
new_list = df_new.values.tolist()

count=0
_list = []
for old_data in old_list:
    name = old_data[0]
    for new_data in new_list:
        if new_data[0] == name:
            if str(new_data[3]) != 'nan':
                count = count + 1
                pre = old_data[1]
                old_data[1] = str(new_data[3])
                print("{}: {} -> {}".format(name, pre, str(new_data[3])))
    _list.append(old_data)

df_new = pd.DataFrame(old_list, columns=['name', 'label', 'text', 'summary', 'description'])

def convert_label(x):
    convert_map = {
    '库': '库',
    '工具' : '工具',
    '服务' : '服务',
    '其它' : '其它',
    '基础环境' : '库',
    }
    return convert_map[x] if x in convert_map else '其它'

df_new['label'] = df_new['label'].apply(lambda x: convert_label(x))

df_new.to_csv("./datasource_0205_class.csv", index=False)


df_new = pd.read_csv("标签调整-20230204.csv")
df_correct = df_new.loc[df_new['调整'] == df_new['pred']]
print("修改与预测相符：", len(df_correct))
print("修改了:：", count)