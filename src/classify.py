import pandas as pd
import numpy as np
import random
import ast
import pickle
import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn import model_selection



def predict(cls, classifier, training_data, labels, test_data, labels_test):
    classifier.fit(training_data, labels)
    results = classifier.predict(test_data)
    print(cls)
    probabilities = classifier.predict_proba(test_data)
    # print(labels_test, results, probabilities)
    print('../results/' + str(cls) + 'site.csv')
    pd.DataFrame(results).to_csv('../results/' + str(cls) + '_temp.csv')
    pd.DataFrame(labels_test).to_csv('../results/' + str(cls) + '_labels_temp.csv')
    # print(labels)
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']
    print(classification_report(labels_test, results, target_names=target_names))
    print(confusion_matrix(labels_test, results))
    print('ROC', roc_auc_score(labels_test, results, average='weighted'))
    # print('Precision:', accuracy_score(labels_test, results))
    # print('F1 score:', f1_score(labels_test, results, average='weighted'))
    # print('Cohen kappa:', cohen_kappa_score(labels_test, results))
    # print('Recall:', recall_score(labels_test, results, average='weighted'))
    return results

def get_data():
    data = pd.read_csv('../features/data_features_clean_flat.csv', header=None)
    return data


def get_features():
    n = 115615
    # n = 500


    labels = pd.read_csv('../features/labels_2_classes.csv').iloc[0:n, 1]
    # features = pd.read_csv('../features/labels_to_test.csv').iloc[0:n, 2:3]
    # embeddings = pd.read_csv('../features/embeddings.csv', encoding='latin1').iloc[0:n, :]
    n_grams = pd.read_csv('../features/ngrams.csv', encoding='latin1').drop('Unnamed: 0', axis=1).iloc[0:n]
    # scores = pd.read_csv('../features/scores.csv', encoding='latin1').drop('Unnamed: 0', axis=1).iloc[0:n, 2:]
    # pos_tags = pd.read_csv('../features/pos_ngrams.csv', encoding='latin1').iloc[0:n]
    # linguistic = pd.read_csv('../features/linguistic.csv', encoding='latin1').iloc[0:n]

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
# print(labels.iloc[0])
# features_reduced = SelectKBest(f_classif, k=3).fit_transform(features, labels)

print('Splitting training and test sets...')
train, test, labels_train, labels_test = train_test_split(
    features, labels,
    test_size=0.20, random_state=42)

print('Starting classification...')
print(datetime.datetime.now())
# predict('Randomm Forest', RandomForestClassifier(n_estimators=1000, n_jobs=-1, verbose=0, class_weight='balanced', bootstrap=True)
#         , train, labels_train, test, labels_test)
# print('Finished Random Forest:', datetime.datetime.now())
predict('Logistic Regression', LogisticRegression(), train, labels_train, test, labels_test)

print('Finished Logistic Regression:', datetime.datetime.now())
