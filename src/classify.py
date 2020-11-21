import pandas as pd
import numpy as np
import random
import ast
import pickle
import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def predict(cls, classifier, training_data, labels, test_data, labels_test):
    classifier.fit(training_data, labels)
    results = classifier.predict(test_data)
    print(cls)
    probabilities = classifier.predict_proba(test_data)
    print('../results/' + str(cls) + 'site.csv')
    pd.DataFrame(results).to_csv('../results/' + str(cls) + '_temp.csv')
    pd.DataFrame(labels_test).to_csv('../results/' + str(cls) + '_labels_temp.csv')
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']
    print(classification_report(labels_test, results, target_names=target_names))
    print(confusion_matrix(labels_test, results))
    print('ROC', roc_auc_score(labels_test, results, average='weighted'))
    print('Precision:', accuracy_score(labels_test, results))
    print('F1 score:', f1_score(labels_test, results, average='weighted'))
    print('Recall:', recall_score(labels_test, results, average='weighted'))
    return results

def get_data(path='../features/data_features_clean_flat.csv'):
    data = pd.read_csv(path, header=None)
    return data


def get_features(base_path='../features/'):
    n = 115615
    # n = 500


    labels = pd.read_csv(base_path + 'labels_2_classes.csv').iloc[0:n, 1]
    # features = pd.read_csv(base_path + 'labels_to_test.csv').iloc[0:n, 2:3]
    # embeddings = pd.read_csv(base_path + 'embeddings.csv', encoding='latin1').iloc[0:n, :]
    n_grams = pd.read_csv(base_path + 'ngrams.csv', encoding='latin1').drop('Unnamed: 0', axis=1).iloc[0:n]
    # scores = pd.read_csv(base_path + 'scores.csv', encoding='latin1').drop('Unnamed: 0', axis=1).iloc[0:n, 2:]
    # pos_tags = pd.read_csv(base_path + 'pos_ngrams.csv', encoding='latin1').iloc[0:n]
    # linguistic = pd.read_csv(base_path + 'linguistic.csv', encoding='latin1').iloc[0:n]

    print('Merging features...')

    # features = embeddings.drop('0', axis=1)
    features = n_grams.drop('rev_id', axis=1)
    # features = scores.merge(linguistic, left_on='rev_id', right_on='0').drop('0', axis=1).drop('rev_id', axis=1)
    # features = linguistic.drop('0', axis=1)
    # features = scores

    # features = embeddings.merge(linguistic, on='0')
    # features = n_grams.merge(features, left_on='rev_id', right_on='0').drop('0', axis=1)
    # # features = embeddings.merge(n_grams, left_on='0', right_on='rev_id')
    # features = features.merge(scores, on='rev_id')
    # features = features.merge(pos_tags, on='rev_id')
    # # features = features.merge(linguistic, left_on='rev_id', right_on='0')

    # print(features.iloc[0])

    return features.values, labels


print('Getting features and labels...')
features, labels = get_features()
# print(labels)
print('Reducing features...')

print('Splitting training and test sets...')
train, test, labels_train, labels_test = train_test_split(
    features, labels,
    test_size=0.20, random_state=42)

classifiers = {'Logistic Regression': LogisticRegression(),
               'Random Forest': RandomForestClassifier(n_estimators=1000, n_jobs=-1, verbose=0, class_weight='balanced', bootstrap=True),
               'Naive Bayes': GaussianNB(),
               'LDA': LDA(),
               'Decision Tree': DecisionTreeClassifier(random_state=0),
               'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=0)
               }


for classifier in classifiers:
    print('Starting classification...')
    predict(classifier, classifiers[classifier], train, labels_train, test, labels_test)
    print('Finished ' + classifier, datetime.datetime.now())
