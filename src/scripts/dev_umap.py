import umap
import umap.plot
import scipy.sparse
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

tfidf = scipy.sparse.load_npz("114_tfidf.npz")
df = pd.read_csv("data/hein-daily/speeches_114.txt", sep="|", encoding="iso-8859-1")
df.shape
tfidf.shape

idx = np.random.randint(tfidf.shape[0], size=1000)

mapper = umap.UMAP(
    metric='hellinger',
    n_neighbors=250,
    min_dist=0
).fit(tfidf[idx])


mapper




df = pd.DataFrame(mapper.embedding_, columns=["UMAP1", "UMAP2"])

f, ax = plt.subplots(1, 1, figsize=(15, 8))
