import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle


def get_data():
    data = pd.read_pickle('../data/data_features_clean.txt')
    return data


def get_sentences():
    sentences = pd.read_csv('../data/sentences_clean.csv', encoding='latin1').drop(['Unnamed: 0'], axis=1)
    return sentences


# TODO - rerun with (2, 2) ngrams, maybe try with word2vec as well, this does not give bigrams
def get_n_grams(n):
    sentences = get_sentences().iloc[:, 0].tolist()
    tfidf = TfidfVectorizer(min_df=0.01, max_df=0.7, ngram_range=(1, n))
    features = tfidf.fit_transform(sentences)
    ngrams = pd.DataFrame(features.todense(), columns=tfidf.get_feature_names())
    ngrams.to_csv('../features/ngrams.csv')
    print(ngrams)
    return ngrams

    # with open("../data/ngrams.txt", "wb") as fp:  # Pickling
    #     pickle.dump(features, fp)


def merge_data():
    ngrams = pd.read_csv('../features/ngrams.csv')
    data = get_data()
    data = data.merge(ngrams, left_index=True, right_index=True).drop('Unnamed: 0', axis=1)
    with open("../data/data_features_clean.txt", "wb") as fp:  # Pickling
        pickle.dump(data, fp)



# data = get_data()
# ngrams = get_n_grams(2)
merge_data()
# print(ngrams)

