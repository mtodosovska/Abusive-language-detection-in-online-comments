import pandas as pd
import numpy as np
import pickle
import re


def clean_re(comment):
    # print('New lines')
    com = re.sub(r'NEWLINE_TOKEN', ' ', comment)
    comm = re.sub(r'\d+', '.', com)
    return comm


def clean_whitespaces(comment):
    # print('Whitespaces')
    comment = re.sub(r'[^\w\s]{2,}', '', comment)
    comment = re.sub(r' [^\w\s] ', ' ', comment)
    comment = re.sub(r' {2,}', ' ', comment)

    return comment


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
    c = 0
    for index, row in partial.iterrows():
        print(index)
        scores = row.iloc[1: length]
        # avg = int(round(np.mean(scores)))
        avg = np.mean(scores)
        av = 0

        if avg < 0.05:
            av = 0
            c += 1
        elif avg < 0.1:
            av = 1
        elif avg < 0.3:
            av = 2
        elif avg < 0.5:
            av = 3
        elif avg < 0.7:
            av = 4
        else:
            av = 5

        avg_partial = avg_partial.append([np.append(int(row.iloc[0]), av)])
    print(c/115863)

    # avg_partial.to_csv('../data/partial_avg.csv')

    avg_partial.to_csv('../features/labels.csv')

    print(avg_partial)

    return avg_partial


def read_data():
    # comments = pd.read_csv('../data/comments_clean.csv', encoding='latin1').drop(['Unnamed: 0'], axis=1)
    with open("../data/comments_clean.txt", "rb") as fp:  # Unpickling
        comments = pickle.load(fp)

    labels = pd.read_csv('../data/4054689/attack_annotations.tsv', delimiter='\t')
    labels = labels.drop(['worker_id', 'quoting_attack', 'recipient_attack', 'third_party_attack', 'other_attack'], axis=1)
    return labels, comments


def merge_data(group, comments):
    data = group.merge(comments, left_on='0', right_on=0, how='inner')
    data.columns = ['rev_id', 'label', '0', 'words']
    data = data.drop(data.columns[2], axis=1)

    with open("../data/data.txt", "wb") as fp:  # Pickling
        pickle.dump(data, fp)
    # data.to_csv('../data/data.csv')
    return data


def merge_comments_words():
    with open("../features/basic.txt", "rb") as fp:  # Unpickling
        basic = pickle.load(fp)
    comments = pd.read_csv('../data/4054689/attack_annotated_comments.tsv', delimiter='\t')
    comments = comments.drop(['year', 'sample', 'split'], axis=1)

    basic_comments = basic.merge(comments, left_on=2, right_on='rev_id')
    labels = basic_comments.iloc[:, 0:2]
    basic_comments = basic_comments.drop(2, axis=1)
    basic_comments = basic_comments.drop(1, axis=1)
    cl = basic_comments.columns.tolist()
    cl[0] = 'comment'
    with open("../features/basic_comments.txt", "wb") as fp:  # Pickling
        pickle.dump(basic_comments, fp)

    labels = labels[[2, 1]]
    cols = ['rev_id', 'label']
    labels.columns = cols
    pd.DataFrame(labels).to_csv('../features/labels.csv', header=cols)


def create_basic_data():
    data = pd.read_pickle('../data/data.txt')
    comments = pd.read_csv('../data/4054689/attack_annotated_comments.tsv', delimiter='\t', encoding='latin1')
    basic = data.merge(comments, on='rev_id')
    # print(basic)
    basic = basic.drop('label', axis=1)
    basic = basic.drop('sample', axis=1)
    basic = basic.drop('split', axis=1)
    # print(basic.iloc[0, :])

    revs = []

    for index, row in basic.iterrows():
        if len(row['words']) == 0:
            revs.append(row['rev_id'])

    print(revs)

    bsc = pd.DataFrame()

    for index, row in basic.iterrows():
        print(index)
        if row['rev_id'] not in revs:
            bsc = bsc.append(row)

    basic = bsc
    print(basic)

    with open("../features/basic_comments.txt", "wb") as fp:  # Pickling
        pickle.dump(basic, fp)


def clean_new_lines():
    data = pd.read_pickle('../features/basic_comments_clean.txt')
    data = data.reset_index(drop=True)
    for index, row in data.iterrows():
        data.iloc[index, 0] = clean_whitespaces(row['comment'])
        if index % 1000 == 0:
            print(index)
            print(row['comment'])
            print(data.iloc[index, 0])

        if index % 10000 == 0:
            print('Bingoooo!')
            with open("../features/basic_comments_clean.txt", "wb") as fp:  # Pickling
                pickle.dump(data, fp)

    print(data)

    # data = data.reset_index(drop=True)
    # for i in range(115000, 115615):
    #     data.iloc[i, 0] = clean_re(data.iloc[i, 0])

    data = data.reset_index(drop=True)
    with open("../features/basic_comments_clean.txt", "wb") as fp:  # Pickling
        pickle.dump(data, fp)




# basic = pd.read_pickle('../features/basic_comments.txt')
# print(basic.iloc[0, :])
# print(basic)
# ids = set(basic.iloc[:, 1])
# print(ids)
# data = pd.read_csv('../data/4054689/attack_annotated_comments.tsv', delimiter='\t', encoding='latin1')
# print(data)
# print(data.iloc[0, :])

# create_basic_data()
# clean_new_lines()
data = pd.read_pickle('../features/basic_comments_clean.txt')
# print(data)

labels = pd.read_csv('../features/labels.csv').drop('Unnamed: 0', axis=1).drop('Unnamed: 0.1', axis=1)

labels.columns = ['rev_id', 'label']
dt = data.merge(labels, how='left', on='rev_id')
dt = dt.drop('year', axis = 1)
dt = dt.drop('logged_in', axis = 1)
dt = dt.drop('comment', axis = 1)
dt = dt.drop('words', axis = 1)
dt = dt.drop('ns', axis = 1)
print(dt)
print(dt.iloc[0])
dt = dt.reset_index()
dt.to_csv('../features/labels.csv', index=False)
