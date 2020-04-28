from collections import Counter
import pandas as pd
import spacy
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import os

nlp = spacy.load("en_core_web_lg")
sw = stopwords.words("english")
ps = PorterStemmer()


class DocumentFeatureMatrix:
    """
    Currently a very inefficient pandas-based implementation of a dfm
    constructor. Advantage of utilizing spacy doc objects, allowing for much
    sophisticated parsing of tokens.
    TODO:
    - Iteratively apply spacy nlp method to reduce memory usage.
    - Parameterize token removal.
    - Add options to store as sparse matrix
    """
    def __init__(self, src, text_col="nlp_prepped", lower_lim=0.01, include_src=False):
        if isinstance(src, pd.DataFrame):
            self.df = src
            self.text_col = src[text_col]
            if isinstance(self.text_col[0], spacy.tokens.doc.Doc):
                self.dfm = self.nlp_col_to_dfm(self.text_col)
            else:
                self.dfm = self.text_col_to_dfm(self.text_col)
            self.reduced_dfm = self.reduce_dfm(self.dfm, lower_lim)
        elif isinstance(src, str):
            self.meta = pd.read_csv(src+"/metadata.csv")
            self.meta.loc[:, 'date'] = pd.to_datetime(self.meta['date'])
            self.dfm = pd.read_csv(src+"/dfm.csv")
            if include_src:
                self.src_dfm = pd.read_csv(src+"/src_dfm.csv")
        else:
            TypeError(
                "DFM constructor only accepts pandas dataframe or pre-loaded directory.")

    def text_col_to_dfm(self, text_series):
        """
        A more efficient function for constructing the dfm.
        """


    def text_col_to_dfm(self, text_series):
        index = text_series.index.values
        tokens = list(set(text_series.str.split(" ").sum()))
        data = []
        print(
            f"Constructing dfm from {len(index)} documents, {len(tokens)} unique tokens.")
        for i in index:
            row = []
            l = Counter(text_series.values[i].split(" "))
            for token in tokens:
                row.append(l.get(token, 0))
            data.append(row)
        df = pd.SparseDataFrame(index=index, columns=tokens, data=data)
        print(f"dfm constructed.")
        return df

    def nlp_col_to_dfm(self, text_series):
        index = text_series.index.values
        exclude_pos_ = ['PUNCT', 'SYM', 'PART', 'SPACE', 'DET']
        exclude_tag_ = ['.', '$', 'DT', '``', "''", ',', 'TO',
                        'POS', 'CD', 'SYM', ':', '-LRB-', '-RRB-',
                        '_SP', 'HYPH', 'LS', 'NFP', 'CC']
        tokens = set(
            text_series.apply(
                lambda doc: [ps.stem(token.text) for token in doc if
                             token.tag_ not in exclude_tag_
                             and token.pos_ not in exclude_pos_
                             and token.text.lower() not in sw]).sum())
        print(
            f"Constructing dfm from {len(index)} documents, {len(tokens)} unique tokens.")
        data = []
        for i in index:
            row = []
            parsed = [ps.stem(token.text) for token in text_series[i]]
            l = Counter(parsed)
            for token in tokens:
                row.append(l.get(token, 0))
            data.append(row)
        df = pd.SparseDataFrame(index=index, columns=tokens, data=data)
        print(f"dfm constructed.")
        df.index.rename('docid')
        return df

    def reduce_dfm(self, dfm, lower_lim):
        """
        Function to reduce dimensionality by removing short tokens and rarely
        occurring ones.
        """
        word_freq = dfm.sum()
        n_features = dfm.shape[1]
        # Remove bottom 1% of words
        reduced_dfm = dfm.loc[:, (word_freq > word_freq.quantile(lower_lim))]
        reduced_dfm = reduced_dfm.drop(
            [t for t in reduced_dfm.columns if len(t) < 2], axis=1)
        print("%d features removed, %0.2f percent" % (
            (n_features - reduced_dfm.shape[1]),
            100*(reduced_dfm.shape[1]/n_features)))
        # Remove words that are truncations of other words.
        return reduced_dfm

    def tfidf(self):
        """
        TODO:
        Implement tf-idf transformation.
        """
        return None

    def save_dfm(self, metadata_cols, path_to_folder="./", overwrite=False):
        folder = f"dfm_{self.reduced_dfm.shape[0]}_{self.reduced_dfm.shape[1]}"
        save_dir = path_to_folder+folder
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        else:
            if not overwrite:
                i = 1
                save_dir = save_dir+"_"+str(i).zfill(2)
                while os.path.exists(save_dir):
                    i += 1
                os.mkdir(save_dir)
            else:
                os.mkdir(save_dir)
        # metadata_cols = ['date', 'name', 'Party', 'Electorate',
        #                  'tweet_username', 'List', 'text', 'prepped']
        self.df[metadata_cols].to_csv(save_dir+"/metadata.csv", index=False,
                                      date_format="%Y/%m/%d %H:%M:%S")
        self.dfm.to_csv(save_dir+"/src_dfm.csv", index=False)
        self.reduced_dfm.to_csv(save_dir+"/dfm.csv", index=False)
        print(f"Saved metadata, source dfm, reduced dfm to {save_dir}.")
