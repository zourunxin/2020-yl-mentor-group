from gensim.models import Word2Vec
import os
import re

from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

'''
    从 build_dataset 复制来修改的，后续合并到一个工具类里面
'''
invalid_word = stopwords.words('english')

# 获取单词的词性


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def is_valid(word):
    if re.match("[()\-:;,.0-9]+", word) or word in invalid_word:
        return False
    elif len(word) < 4:
        return False
    else:
        return True


def load_corpus():
    f1 = open('../resources/raw_features.txt', encoding='gbk')
    raw_lines = []
    for line in f1:
        raw_lines.append(line.strip())

    raw_datas = []
    raw_data = []

    description = ""

    n = 0
    while (n < len(raw_lines)):
        if (raw_lines[n].split(':')[0] == 'depname'):
            raw_data = {}
            n += 1
        while (raw_lines[n].split(':')[0].strip() != 'Description'):
            raw_data[raw_lines[n].split(':', 1)[0].strip()] = raw_lines[n].split(':', 1)[
                1].strip()
            n += 1
        if (raw_lines[n].split(':')[0].strip() == 'Description'):
            description = ""
            n += 1
            while (n < len(raw_lines) and raw_lines[n].split(':')[0] != 'depname'):
                description += raw_lines[n].strip() + ' '
                n += 1
            raw_data['Description'] = description
            raw_datas.append(raw_data)
    sentences = []

    for data in raw_datas:
        sentences.append(data['Description'])
        sentences.append(data['Summary'])

    finalSentences = []

    for sentence in sentences:
        # print(sentence)
        tokens = word_tokenize(sentence)  # 分词
        tagged_sent = pos_tag(tokens)  # 获取单词词性

        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            word = wnl.lemmatize(tag[0], pos=wordnet_pos)  # 词形还原
            if (is_valid(word)):
                lemmas_sent.append(word)
        finalSentences.append(lemmas_sent)

    return finalSentences


'''
    使用所有包的 summary 和 description 信息建立语料库， 训练 word2vec
'''
model_path = './word2vec_model.pkl'

if not os.path.exists(model_path):
    print('正在建立语料库')
    corpus = load_corpus()
    model = Word2Vec(corpus, vector_size=100, window=5, min_count=5)
    model.save(model_path)

print('正在加载语料库')
model = Word2Vec.load(model_path)

# 测试
y2 = model.wv.most_similar("library", topn=5)  # 5个最相关的
print("the most similar word to library: ")
for item in y2:
    print(item[0], item[1])
print("--------\n")
