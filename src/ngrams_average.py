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


def get_n_grams(n):

    with open("../features/basic_comments.txt", "rb") as fp:  # Unpickling
        basic = pickle.load(fp)

    # print(basic)

    # sentences = []
    # i = 0
    # for comment in basic['words']:
    #     print('Sentences:', i)
    #     i += 1
    #     sentences.append(" ".join(str(x) for x in comment))
    #
    # sentences = pd.DataFrame(sentences)
    # sentences.to_csv('../data/sentences_clean.csv')
    sentences = pd.read_csv('../data/sentences_clean.csv', encoding='latin1').drop('Unnamed: 0', axis=1)
    # sentences = basic['words']
    # print(sentences)

    tfidf = TfidfVectorizer(max_features=1500, ngram_range=(2, n))
    features = tfidf.fit_transform(sentences.iloc[:, 0])
    ngrams = pd.DataFrame(features.todense(), columns=tfidf.get_feature_names())

    # ngrams = ngrams.merge(basic['rev_id'], left_index=True, right_index= True)
    bsc = basic.drop('logged_in', axis=1)\
        .drop('ns', axis=1)\
        .drop('comment', axis=1)\
        .drop('words', axis=1)

    ngrams = ngrams.reset_index(drop=True)
    bsc = bsc.reset_index(drop=True)

    print(bsc)

    bsc = bsc.merge(ngrams, left_index=True, right_index=True)\
        .drop('year', axis=1)

    bsc.to_csv('../features/ngrams3.csv')
    print(ngrams)
    print(bsc)
    # return ngrams

    # with open("../data/ngrams.txt", "wb") as fp:  # Pickling
    #     pickle.dump(features, fp)


def merge_data():
    ngrams = pd.read_csv('../features/ngrams.csv')
    data = get_data()
    data = data.merge(ngrams, left_index=True, right_index=True).drop('Unnamed: 0', axis=1)
    with open("../data/data_features_clean.txt", "wb") as fp:  # Pickling
        pickle.dump(data, fp)


# data = get_data()
get_n_grams(3)
# merge_data()
# print(ngrams)

