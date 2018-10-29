import pandas as pd
import pickle
import math
import numpy as np
import re
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



def get_data():
    data = pd.read_pickle('../data/data_features_clean.txt')
    # data = data[['rev_id', 'label', 'words', 'embedding']]
    return data


def delete_empty_rows(data):
    i = 0
    new_data = pd.DataFrame()
    for index, row in data.iterrows():
        print('To delete or not to delete the question is now:', i)
        if len(list(row.iloc[2])) != 0:
            new_data = new_data.append(row)
        i += 1

    print(new_data)
    with open("../data/data_features_clean.txt", "wb") as fp:  # Pickling
        pickle.dump(new_data, fp)
    return new_data


def get_sentences():
    sentences = pd.read_csv('../data/sentences.csv', encoding='latin1').drop(['Unnamed: 0'], axis=1)
    return sentences


def clean_sentences():
    sentences = get_sentences()
    i = 0
    new_data = pd.DataFrame()
    for index, row in sentences.iterrows():
        if i % 1000 == 0:
            print('To delete or not to delete the question is now:', i)
        if isinstance(row[0], float) and math.isnan(row[0]):
            print('Deleting...', i)
        else:
            new_data = new_data.append(row)
        i += 1

    new_data.to_csv('../data/sentences_clean.csv')
    print(new_data)


def get_embeddings():
    data = pd.read_pickle('../data/data_features_clean.txt')
    data = data[['rev_id', 'label', 'words', 'embedding']]
    data.set_index('rev_id')
    embeddings = data['embedding']
    with open("../features/embeddings.txt", "wb") as fp:  # Pickling
        pickle.dump(embeddings, fp)


def flat_data(data):
    embeddings = data['embedding'].tolist()
    data = data.drop(['embedding'], axis=1)
    data = data.reset_index(drop=True)
    names = list(data)
    embeddings_names = [str('embedding: ' + str(x + 1)) for x in range(0, 100)]
    # names.append(embeddings_names)
    names[len(names):len(names) + 1] = embeddings_names
    print(names)
    new_data = pd.DataFrame()

    for index, row in data.iterrows():
        print(index)
        row = pd.DataFrame(np.append(row.values, embeddings[index])).transpose()
        new_data = new_data.append(row)


    new_data = new_data.reset_index(drop=True)
    # new_data = new_data.drop(0, axis=1)
    # print(new_data)?
    # new_data.rename(columns={1: 'rev_id'}, inplace=True)
    new_data.columns = names

    print(new_data)
    new_data.to_csv('../features/data_features_clean_flat.csv', header=False, mode='a')
    # with open("../features/data_features_clean_flat.txt", "wb") as fp:  # Pickling
    #     pickle.dump(new_data, fp)


def flat_data_calling():
    for i in range (0, 115):
        print('-----------', i)
        data = get_data()
        dt = data.iloc[1000*i:1000*(i+1)]
        del data
        flat_data(dt)

    data = get_data()
    dt = data.iloc[1000*i:]
    del data
    flat_data(dt)


def separate_features():
    data = pd.read_csv('../features/data_features_clean_flat.csv', header=None).drop(0, axis=1)
    print(data)
    basic = data.iloc[:, 1:3]
    data = data.drop(3, axis=1)
    # n_grams = data.iloc[:, 1:525]
    for i in range(4, 527):
        data = data.drop(i, axis=1)
    data = data.drop(1, axis=1)

    embeddings = data
    print('basic', basic)
    # print('n_grams', n_grams)
    print('embeddings', embeddings)
    # n_grams.to_csv('../features/n_grams.csv')
    embeddings.to_csv('../features/embeddings.csv')
    with open("../features/basic.txt", "wb") as fp:  # Pickling
        pickle.dump(basic, fp)


def sentences_clean():
    data = pd.read_pickle('../features/basic_comments_clean.txt')
    comments = data.drop('logged_in', axis=1) \
        .drop('ns', axis=1) \
        .drop('year', axis=1) \
        .drop('comment', axis=1)
    del data
    comments = comments[['rev_id', 'words']]
    sentences = pd.read_csv('../data/sentences_clean.csv', encoding='latin1').drop('Unnamed: 0', axis=1)
    sent = comments.merge(sentences, how='left', left_index=True, right_index=True).drop('words', axis=1)
    sent.columns = ['rev_id', 'sentence']
    print(sent)
    # print(sent.iloc[0])
    sent.to_csv('../features/sentences_clean.csv', index=False)


def relabel():
    labels = pd.read_csv('../features/labels.csv').iloc[:, :]
    class_0 = labels[labels['label'] == 0].iloc[0:, :]
    class_2 = labels[labels['label'] == 2].append(labels[labels['label'] == 1]).iloc[0:, :]
    class_2.iloc[:, 2] = 0
    class_3 = labels[labels['label'] == 3].append(labels[labels.label == 4]).append(labels[labels.label == 5]).iloc[0:,:]
    class_3.iloc[:, 2] = 1
    print(class_0.shape)
    print(class_2.shape)
    # size = int((class_3.shape[0])/2)
    # class_0 = class_0.sample(frac=1).iloc[0:size, :]
    # class_2 = class_2.sample(frac=1).iloc[0:size, :]
    labels = class_0.append(class_2).append(class_3)

    labels = labels.sample(frac=1)
    labels = labels.reset_index().drop('level_0', axis=1).drop('index', axis=1)
    print(labels)
    labels.to_csv('../features/labels_2_classes.csv', index=False)
    print('-------------Labels done!++++++++++++++')

def get_balanced_labels():
    labels = pd.read_csv('../features/labels_3_classes.csv').iloc[:, :]
    labels_3 = labels[labels.label == 3]
    labels_0 = labels[labels.label == 0]
    labels_0 = labels_0.sample(frac=1).iloc[0:labels_3.shape[0]]
    labels_2 = labels[labels.label == 2]
    labels_2 = labels_2.sample(frac=1).iloc[0:labels_3.shape[0]]
    # labels_3 = labels_3.sample(frac=1)
    labels_all = labels_0.append(labels_2).append(labels_3)
    print(labels_all)
    return labels_all


def clean_completely():
    basic = pd.read_pickle('../features/basic_comments_clean.txt')
    print(basic['comment'][0])
    basic['comment'] = basic['comment'].apply(lambda x: x.lower())
    basic['comment'] = basic['comment'].apply(lambda x: re.sub(r'\W', ' ', x))
    basic['comment'] = basic['comment'].apply(lambda x: re.sub(r' {2,}', ' ', x))
    print(basic['comment'][0])
    basic.to_pickle('../features/basic_comments_tokenised.txt')


def check_evaluation_3nb():
    labels = np.zeros(23123)
    results = np.zeros(23123)
    labels[0:9695] = 0
    results[0:9695] = 0

    labels[9695:11870] = 1
    results[9695:11870] = 1

    labels[11870:14498] = 2
    results[11870:14498] = 2

    labels[14498:15910] = 0
    results[14498:15910] = 1

    labels[15910:16220] = 0
    results[15910:16220] = 2

    labels[16220:16896] = 1
    results[16220:16896] = 2

    labels[16896:20806] = 1
    results[16896:20806] = 0

    labels[20806:21900] = 2
    results[20806:21900] = 0

    labels[21900:] = 2
    results[21900:] = 1

    print(confusion_matrix(labels, results))
    # print(roc_auc_score(labels, results))
    print(classification_report(labels, results))
    print('Precision:', precision_score(labels, results, average='macro'))
    print('Recall:', recall_score(labels, results, average='macro'))
    print('F1:', f1_score(labels, results, average='macro'))


def check_evaluation_3b():
    labels = np.zeros(14996)
    results = np.zeros(14996)
    labels[0:3161] = 0
    results[0:3161] = 0

    labels[3161:5658] = 1
    results[3161:5658] = 1

    labels[5658:8855] = 2
    results[5658:8855] = 2

    labels[8855:10356] = 0
    results[8855:10356] = 1

    labels[10356:10647] = 0
    results[10356:10647] = 2

    labels[10647:12305] = 1
    results[10647:12305] = 0

    labels[12305:13249] = 1
    results[12305:13249] = 2

    labels[13249:13672] = 2
    results[13249:13672] = 0

    labels[13672:] = 2
    results[13672:] = 1

    print(confusion_matrix(labels, results))
    # print(roc_auc_score(labels, results))
    print(classification_report(labels, results))
    print('Precision:', precision_score(labels, results, average='macro'))
    print('Recall:', recall_score(labels, results, average='macro'))
    print('F1:', f1_score(labels, results, average='macro'))


def check_evaluation_6nb():
    labels = np.zeros(23123)
    results = np.zeros(23123)

    labels[0:10084] = 0
    results[0:10084] = 0

    labels[10084:10094] = 1
    results[10084:10094] = 1


    labels[10184:12066] = 2
    results[10184:12066] = 2

    labels[12066:12226] = 3
    results[12066:12226] = 3

    labels[12226:12336] = 4
    results[12226:12336] = 4

    labels[12336:13471] = 5
    results[12336:13471] = 5

    labels[13471:13572] = 0
    results[13471:13572] = 1

    labels[13572:14695] = 0
    results[13572:14695] = 2

    labels[14695:14775] = 0
    results[14695:14775] = 3

    labels[14775:14839] = 0
    results[14775:14839] = 4

    labels[14839:14895] = 0
    results[14839:14895] = 5

    labels[14895:15573] = 1
    results[14895:15573] = 0

    labels[15573:15717] = 1
    results[15573:15717] = 2

    labels[15717:15727] = 1
    results[15717:15727] = 3

    labels[15727:15731] = 1
    results[15727:15731] = 4

    labels[15731:15736] = 1
    results[15731:15736] = 5

    labels[15736:19186] = 2
    results[15736:19186] = 0

    labels[19186:19211] = 2
    results[19186:19211] = 1

    labels[19211:19420] = 2
    results[19211:19420] = 3

    labels[19420:19551] = 2
    results[19420:19551] = 4

    labels[19551:19621] = 2
    results[19551:19621] = 5

    labels[19621:20328] = 3
    results[19621:20328] = 0

    labels[20328:20331] = 3
    results[20328:20331] = 1

    labels[20331:21139] = 3
    results[20331:21139] = 2

    labels[21139:21252] = 3
    results[21139:21252] = 4

    labels[21252:21342] = 3
    results[21252:21342] = 5

    labels[21342:21653] = 4
    results[21342:21653] = 0

    labels[21653:21654] = 4
    results[21653:21654] = 1

    labels[21654:22082] = 4
    results[21654:22082] = 2

    labels[22082:22200] = 4
    results[22082:22200] = 3

    labels[22200:22369] = 4
    results[22200:22369] = 5

    labels[22369:22676] = 5
    results[22369:22676] = 0

    labels[22676:22677] = 5
    results[22676:22677] = 1

    labels[22677:22988] = 5
    results[22677:22988] = 2

    labels[22988:23070] = 5
    results[22988:23070] = 3

    labels[10094:10184] = 5
    results[10094:10184] = 4

    labels[23070:] = 5
    results[23070:] = 4


    print(confusion_matrix(labels, results))
    # print(roc_auc_score(labels, results))
    print(classification_report(labels, results))
    print('Precision:', precision_score(labels, results, average='macro'))
    print('Recall:', recall_score(labels, results, average='macro'))
    print('F1:', f1_score(labels, results, average='macro'))


check_evaluation_6nb()
# basic = pd.read_pickle('../features/basic_comments_clean.txt')
# clean_completely()
# separate_features()
# sentences_clean()
# relabel()