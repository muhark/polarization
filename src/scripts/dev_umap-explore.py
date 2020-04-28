import umap
import umap.plot
import scipy.sparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_style("darkgrid")

embeddings = pd.read_feather("umap_test.feather")
source = pd.read_csv("data/hein-daily/speeches_114.txt", sep="|", encoding="iso-8859-1")
mappings = pd.read_csv("data/hein-daily/114_SpeakerMap.txt", sep="|", encoding="iso-8859-1")

df = pd.merge(
    source,
    mappings,
    how="inner",
    on="speech_id"
)


df = pd.merge(
    df,
    embeddings,
    how="inner",
    left_index=True,
    right_index=True
)

df.columns


df.loc[:, 'party'] = df['party'].fillna("Unknown")

cov


i=0
j=0
f, ax = plt.subplots(2, 2, figsize=(30, 30))
for cov in ["chamber", "state", "gender", "party"]:
    c_dict = dict(zip(df[cov].unique(), sns.color_palette(n_colors=len(df[cov].unique()), palette="viridis")))
    ax[i][j].set_title(f"umap shaded by {cov}")
    scatter = ax[i][j].scatter(
        x=df['UMAP1'], y=df["UMAP2"], s=0.1,
        c=df[cov].apply(lambda x: c_dict[x])
    )
    ax[i][j].set(xticks=[], yticks=[], facecolor='black')
    legend = ax[i][j].legend(*scatter.legend_elements(), loc="upper right")
    ax[i][j].add_artist(legend)
    i+=1
    if i==2:
        i=0


i=0
j=0
f, ax = plt.subplots(2, 2, figsize=(30, 30))
f.suptitle("UMAP Plots for 114th Congress")
f.subplots_adjust(hspace=0.05, top=0.95)
for cov in ["chamber", "state", "gender", "party"]:
    c_dict = dict(zip(df[cov].unique(), sns.color_palette(n_colors=len(df[cov].unique()), palette="viridis")))
    ax[i][j].set_title(f"umap shaded by {cov}")
    sns.scatterplot(
        x='UMAP1', y="UMAP2",
        s=0.1, hue=cov,
        edgecolor=None, data=df,
        palette="husl",
        ax=ax[i][j]
    )
    ax[i][j].set(xticks=[], yticks=[], facecolor='black')

    i+=1
    if i==2:
        i=0
        j+=1

f.savefig("umap_114.png", facecolor="grey", edgecolor=None, bbox_inches="tight")
