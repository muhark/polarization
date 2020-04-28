import os
import re
import umap
import umap.plot
import pandas as pd
import scipy.sparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text

# I suspect that setting n_neighbours to a high value and
# min_dist to a low value will improve visuals.

def tfidf_constructor(file, save_dir, congress):
    print("Constructing tf-idf.")
    speeches = pd.read_csv(file, sep="|", encoding="iso-8859-1")
    vectorizer  = text.TfidfVectorizer()
    tfidf = vectorizer.fit_transform(speeches['speech'])
    features = pd.Series(data=vectorizer.vocabulary_).sort_values().reset_index()
    features.columns = ["token", "index"]
    print("tf-idf constructed: {} documents and {} features.".format(
        tfidf.shape[0],
        tfidf.shape[1]
    ))
    if save_dir:
        scipy.sparse.save_npz(f"{save_dir}/{congress}_tfidf.npz", tfidf)
        features.to_feather(f"{save_dir}/{congress}_features.feather")
    del features
    return tfidf


def umap_fit(tfidf, save_dir, congress, metric="hellinger", n_neighbours=250, min_dist=0):
    print("Initialising UMAP.")
    mapper = umap.UMAP(
        metric=metric,
        n_neighbors=n_neighbours,
        min_dist=min_dist,
        low_memory=True
    ).fit_transform(tfidf)
    print("UMAP fit to tf-idf.")
    embeddings = pd.DataFrame(
        mapper.embedding_,
        columns=["UMAP1", "UMAP2"]
    ).to_feather(f"{save_dir}/{congress}_embeddings.feather")
    if save_dir:
        embeddings.to_feather(f"{save_dir}/{congress}_embeddings.feather")
    del mapper
    return embeddings


def get_covariates(embeddings, read_dir, congress, method="inner"):
    print("Constructing joint table.")
    source = pd.read_csv(f"{read_dir}/speeches_{congress}.txt", sep="|", encoding="iso-8859-1")
    mappings = pd.read_csv(f"{read_dir}/{congress}_SpeakerMap.txt", sep="|", encoding="iso-8859-1")
    df = pd.merge(
        source,
        mappings,
        how="inner",
        on="speech_id"
    )
    df =     pd.merge(
        df,
        embeddings,
        how="inner",
        left_index=True,
        right_index=True
    )
    return df


def plot_umap(df, save_dir, congress):
    print("Generating Plot")
    i=0
    j=0
    f, ax = plt.subplots(2, 2, figsize=(30, 30))
    f.suptitle("UMAP Plots for {congress}th Congress")
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

    f.savefig(f"{save_dir}/umap_{congress}.png", facecolor="grey", edgecolor=None, bbox_inches="tight")

if __name__=="__main__":
    read_dir = "../../data/hein-daily" # Assign
    save_dir = "umap_batch"
    files = os.listdir(read_dir)
    files = [read_dir+"/"+file for file in files if re.match(
            re.compile(r"speeches_[0-9]{3}"), file
        )]
    for file in files:
        try:
            congress = re.search(re.compile(r"[0-9]{3}"), file.split("/")[-1])[0]
            print(f"Beginning process for congress {congress}.")
            tfidf = tfidf_constructor(file, save_dir, congress)
            embeddings = umap_fit(tfidf, save_dir, congress)
            df = get_covariates(embeddings, read_dir, congress)
            plot_umap(df, save_dir, congress)
        except pd.errors.ParserError as e:
            print("Bad input file. Skipping.")
            continue
