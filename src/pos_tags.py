# First create pos tags for the whole dataset and save them in rev_id, pos_tag form
# use the cleaned sentences for this
# Then, use this to create n-grams (trigrams and (maybe) bigrams) for the pos_tags_sentencs
# Side note: the exact same way that you created them for the normal senteces
# and then use these as features

from textblob import TextBlob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def get_comments():
    data = pd.read_csv('../features/sentences_clean.csv')
    return data


def get_pos_tag_comments(comments):

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
    tags.to_pickle('../features/pos_tags.csv')

    print(tags)


def get_n_grams(n):

    sentences = pd.read_pickle('../features/pos_tags.csv')

    sentences_stitch = []
    for i, s in sentences.iterrows():
        sentences_stitch.append(" ".join(str(x) for x in sentences.iloc[i, 1]))
    sentences_stitch

    tfidf = TfidfVectorizer(min_df=0.1, max_df=0.5, ngram_range=(2, n))
    features = tfidf.fit_transform(sentences_stitch)
    ngrams = pd.DataFrame(features.todense(), columns=tfidf.get_feature_names())

    ngrams = ngrams.reset_index(drop=True)

    print(ngrams)
    ngrams.to_csv('../features/pos_ngrams.csv')


def get_ids():
    ngrams = pd.read_csv('../features/pos_ngrams.csv', encoding='latin1')
    scores = pd.read_csv('../features/scores.csv', encoding='latin1').drop('Unnamed: 0', axis=1)
    nngrams = scores.merge(ngrams, how='left', left_index=True, right_index=True)
    nngrams.drop('readability', axis=1) \
            .drop('polarity', axis=1)\
            .drop('subjectivity', axis=1)

    nngrams = nngrams.reset_index()
    nngrams.to_csv('../features/pos_ngrams.csv')

# words = get_comments()
# get_pos_tag_comments(words)
# get_n_grams(3)
get_ids()
