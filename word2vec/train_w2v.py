import sys
sys.path.append("../")
import os
import pandas as pd
import utils.NLPUtils as NLPUtils
from gensim.models.word2vec import Word2Vec

df_data = pd.read_csv("../output/datasource_1228_new.csv")

texts = df_data["text"].values.tolist()

sentences = [NLPUtils.preprocess_text(text).split(" ") for text in texts]

print(len(sentences))
print(sentences[0])

num_features = 128
model = Word2Vec(sentences, vector_size=num_features)
model_name = "word2vec_model_{}d.model".format(num_features)
model.init_sims(replace=True)
model.save(os.path.join("../saved_models",model_name))

print(model.wv.most_similar("library"))
print(model.wv.doesnt_match(["library", "api", "module", "tool"]))