import pandas as pd
import numpy as np
import os
import enchant
import Levenshtein as Lv
from scipy.sparse import save_npz
from sklearn.feature_extraction import text


# RAMSIZE = 31200000000
# SAFE_LIMIT = RAMSIZE*0.75
# SAFE_LIMIT/322608 72533.84912959381

step = 100

df = pd.read_csv("data/hein-daily/speeches_105.txt", sep="|", encoding="iso-8859-1")
vectorizer  = text.TfidfVectorizer()
tfidf = vectorizer.fit_transform(df['speech'])

vocab = pd.Series(vectorizer.vocabulary_).reset_index()
vocab.columns = ['token', 'index']

d = enchant.Dict("en_US")


vocab['is_word'] = vocab['token'].apply(lambda x: d.check(x))

vocab.shape[0]
vocab['is_word'].sum()


vocab['suggest'] = None
vocab.loc[vocab['is_word']==False, 'suggest'] = vocab.loc[vocab['is_word']==False, 'token'].apply(lambda x: d.suggest(x))

vocab.loc[vocab['is_word']==False, ['token', 'suggest']]

def ocr_correction(token):
    """
    A series of methods to probabilistically correct OCR errors.
    Based on enchant autocorrect suggestions plus Levenshtein distance.
    """


# path = 'speeches_114.h5'
# if os.path.exists(path):
#     os.remove(path)
#
# store = pd.HDFStore(path)

# for i in range(step, tfidf.shape[0]+1, step):
#     temp = pd.DataFrame(
#         tfidf[i-step:i, :].todense(),
#         dtype=np.float64,
#         index=pd.RangeIndex(i-step, i, 1))
#     if i == step:
#         store.put('tfidf', temp, format='fixed', expectednrows=step)
#     else:
#         store.append('tfidf', temp, format='fixed', expectednrows=step)
#
# a = scipy.sparse.load_npz("test.npz")
