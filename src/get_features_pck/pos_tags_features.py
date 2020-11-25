# First create pos tags for the whole dataset and save them in rev_id, pos_tag form
# use the cleaned sentences for this
# Then, use this to create n-grams (trigrams and (maybe) bigrams) for the pos_tags_sentencs
# Side note: the exact same way that you created them for the normal senteces
# and then use these as features

from textblob import TextBlob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from data_manager import DataManager


def get_pos_tag_comments(comments, path):

    tags = pd.DataFrame()
    for index, row in comments.iterrows():
        print(index)
        blob = TextBlob(row[1])
        pos = [i[1] for i in blob.tags]

        ls = []
        ls.append(row[0])
        ls.append(pos)
        ft = pd.DataFrame(ls)

        tags = tags.append(ft.transpose())

    tags.columns = ['rev_id', 'tags']
    tags = tags.reset_index(drop=True)
    tags.to_pickle(path)

    print(tags)


def get_n_grams(n, path, path_out):

    sentences = pd.read_pickle(path)

    sentences_stitch = []
    for i, s in sentences.iterrows():
        sentences_stitch.append(" ".join(str(x) for x in sentences.iloc[i, 1]))

    tfidf = TfidfVectorizer(min_df=0.1, max_df=0.5, ngram_range=(2, n))
    features = tfidf.fit_transform(sentences_stitch)
    ngrams = pd.DataFrame(features.todense(), columns=tfidf.get_feature_names())

    ngrams = ngrams.reset_index(drop=True)

    print(ngrams)
    ngrams.to_csv(pat_out)


def get_ids(path):
    ngrams = DataManager.get_pos_ngrams()
    scores = DataManager.get_scores()
    nngrams = scores.merge(ngrams, how='left', left_index=True, right_index=True)
    nngrams.drop('readability', axis=1) \
            .drop('polarity', axis=1)\
            .drop('subjectivity', axis=1)

    nngrams = nngrams.reset_index()
    nngrams.to_csv(path)


def get_pos_tags():
    words = DataManager.get_sentences_clean()
    get_pos_tag_comments(words, '../features/pos_tags.csv')
    get_n_grams(3, '../features/pos_tags.csv', '../features/pos_ngrams.csv')
    get_ids('../features/pos_ngrams.csv')
