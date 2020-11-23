import pandas as pd

import numpy as np
import re
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from data_manager import DataManager


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


check_evaluation_3b()
check_evaluation_6nb()
clean_completely()
