'''
计算每个分类的 top 关键词(采用TF-IDF)
'''
import sys
sys.path.append("../")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import heapq
import utils.NLPUtils as NLPUtils


df_source_data = pd.read_csv("../output/datasource_1228.csv")
data_list = df_source_data.values.tolist()

corpus_map = {}

for data in data_list:
    if not data[2] in corpus_map:
        corpus_map[data[2]] = ""
    corpus_map[data[2]] = corpus_map[data[2]] + " " + data[3]

corpus_list = []
corpus = []

# 保持两个列表顺序一致
for key in corpus_map:
    text = NLPUtils.preprocess_text(corpus_map[key])
    corpus_list.append([key, text])
    corpus.append(text)

vectorizer = CountVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(corpus)
# 查看输出
word_list = vectorizer.get_feature_names_out()
matrix = matrix.toarray()

output = []
for i, class_text in enumerate(corpus_list):
    line = matrix[i]
    max_idx = heapq.nlargest(30, range(len(line)), line.take)
    words = []
    for idx in max_idx:
        words.append(word_list[idx])
    output.append([corpus_list[i][0], words])

tf_df = pd.DataFrame(output, columns=['label', 'word_list'])
tf_df.to_csv("../output/count类别特征词.csv")