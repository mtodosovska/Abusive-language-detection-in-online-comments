import pandas as pd
import nltk
import numpy as np
from textblob import TextBlob


def get_data():
    data = pd.read_csv('../features/data_features_clean_flat.csv', header=None)
    return data


def calculate_scores(basic):
    scores = pd.DataFrame()
    for index, row in basic.iterrows():

        print(index)
        comment = row[1]
        blob = TextBlob(comment)

        sentences = len(TextBlob(comment))
        words = sum([len(x.words) for x in blob.sentences])
        characters = len(str(comment))

        readability = 4.71 * (characters / words) + 0.5 * (words / sentences) - 21.43

        sen_pol = blob.sentiment
        polarity = sen_pol[0]
        subjectivity = sen_pol[1]

        ls = np.zeros(4)
        ls[0] = row[0]
        ls[1] = readability
        ls[2] = polarity
        ls[3] = subjectivity
        ft = pd.DataFrame(ls)

        scores = scores.append(ft.transpose())

    scores.columns = ['rev_id', 'readability', 'polarity', 'subjectivity']
    print(scores)
    scores.to_csv('../features/scores.csv')


def get_comments():
    data = pd.read_pickle('../features/basic_comments_clean.txt')
    comments = data.drop('logged_in', axis=1)\
        .drop('ns', axis=1)\
        .drop('year', axis=1)\
        .drop('words', axis=1)
    comments = comments[['rev_id', 'comment']]
    return comments


comments = get_comments()
calculate_scores(comments)
