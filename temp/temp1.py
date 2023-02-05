import sys
sys.path.append("../")
import pandas as pd
import utils.NLPUtils as NLPUtils

df_data = pd.read_csv("../output/datasource_0117_class.csv")
df_word = pd.read_csv("aa.csv")

word_list = [word[0] for word in df_word.values.tolist()]


df_data["text"] = df_data["text"].apply(lambda x: NLPUtils.preprocess_text(x))
data_list = df_data.values.tolist()
result_map = {}

for word in word_list:
    result_map[word] = {}
    for data in data_list:
        if not data[1] in result_map[word]:
            result_map[word][data[1]] = 0
        if data[2].find(word) != -1:
            result_map[word][data[1]] = result_map[word][data[1]] + 1
    print("finish")
        
print(result_map)

for key in result_map:
    for label in result_map[key]:
        result_map[key][label] = result_map[key][label] / len(df_data[df_data["label"] == label]) * 100

print(result_map)

