import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

from data_manager import DataManager


def get_n_grams(path):

    basic = DataManager.get_basic_comments()
    sentences = DataManager.get_sentences_clean()

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    features = tfidf.fit_transform(sentences.iloc[:, 0])
    ngrams = pd.DataFrame(features.todense(), columns=tfidf.get_feature_names())

    bsc = basic.drop('logged_in', axis=1)\
        .drop('ns', axis=1)\
        .drop('comment', axis=1)\
        .drop('words', axis=1)

    ngrams = ngrams.reset_index(drop=True)
    bsc = bsc.reset_index(drop=True)

    bsc = bsc.merge(ngrams, left_index=True, right_index=True)\
        .drop('year', axis=1)

    bsc.to_csv(path)


def merge_data(path):
    ngrams = DataManager.get_ngrams()
    data = DataManager.get_data_features_clean()
    data = data.merge(ngrams, left_index=True, right_index=True).drop('Unnamed: 0', axis=1)
    with open(path, "wb") as fp:  # Pickling
        pickle.dump(data, fp)


get_n_grams('../features/ngrams_check.csv')
merge_data("../data/data_features_clean.txt")
