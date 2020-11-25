import pandas as pd
import numpy as np
from textblob import TextBlob
from data_manager import DataManager


def get_data():
    data = DataManager.get_data_features()
    return data


def calculate_scores(basic, scores_path):
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
    scores.to_csv(scores_path)


def get_scores():
    comments = DataManager.get_comments()
    calculate_scores(comments, '../features/scores.csv')
