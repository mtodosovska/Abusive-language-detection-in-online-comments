import pandas as pd
import numpy as np
import pickle

def group_data(labels):
    grouped = labels.groupby('rev_id')
    i = 0
    partial = pd.DataFrame()
    print('Total:', len(grouped))
    for name, group in grouped:
        print(i)
        temp = pd.DataFrame(np.zeros(group.shape[0] + 1))
        temp.iloc[0] = group.iloc[0, 0]
        j = 0
        for index, row in group.iterrows():
            temp.iloc[j + 1] = row[1]
            j += 1
        partial = partial.append(temp.transpose())
        i += 1

    print(partial)
    partial.to_csv('../data/partial.csv')


def get_average_scores():
    partial = pd.read_csv('../data/partial.csv').drop(['Unnamed: 0'], axis=1)
    avg_partial = pd.DataFrame()

    length = partial.iloc[0, :].shape[0]

    for index, row in partial.iterrows():
        print(index)
        scores = row.iloc[1: length]
        avg = int(round(np.mean(scores)))
        avg_partial = avg_partial.append([np.append(int(row.iloc[0]), avg)])

    avg_partial.to_csv('../data/partial_avg.csv')
    with open("../data/partial_avg.txt", "wb") as fp:  # Pickling
        pickle.dump(avg_partial, fp)
    return avg_partial


def read_data():
    # comments = pd.read_csv('../data/comments_clean.csv', encoding='latin1').drop(['Unnamed: 0'], axis=1)
    with open("../data/comments_clean.txt", "rb") as fp:  # Unpickling
        comments = pickle.load(fp)


    labels = pd.read_csv('../data/4054689/attack_annotations.tsv', delimiter='\t')
    labels = labels.drop(['worker_id', 'quoting_attack', 'recipient_attack', 'third_party_attack', 'other_attack'], axis=1)
    return labels, comments


def read_group_data():
    data = pd.read_csv('../data/partial_avg.csv', index_col=0)
    return data


def merge_data(group, comments):
    data = group.merge(comments, left_on='0', right_on=0, how='inner')
    data.columns = ['rev_id', 'label', '0', 'words']
    data = data.drop(data.columns[2], axis=1)

    with open("../data/data.txt", "wb") as fp:  # Pickling
        pickle.dump(data, fp)
    # data.to_csv('../data/data.csv')
    return data


def get_merged_data():
    data = pd.read_csv('../data/data.csv', index_col=0)
    return data

def get_labels():
    partial = get_average_scores()
    data = merge_data()


labels, comments = read_data()
partial = read_group_data()
# partial = get_average_scores()
data = merge_data(partial, comments)
print(data)
# group = read_group_data()
# data = get_merged_data()


