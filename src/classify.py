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
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from sklearn.base import BaseEstimator, TransformerMixin

def predict_randomForest(training_data, labels, test_data, labels_test):
    b_rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, verbose=0, class_weight='balanced', bootstrap=True)
    print('Fitting..')
    b_rf.fit(training_data, labels)
    results = b_rf.predict(test_data)
    probabilities = b_rf.predict_proba(test_data)
    with open("../results/random_forest_results.txt", "wb") as fp:  # Pickling
        pickle.dump(results, fp)
    with open("../results/random_forest_probabilities.txt", "wb") as fp:  # Pickling
        pickle.dump(results, fp)
    print("Random forest")
    print(accuracy_score(labels_test, results))
    return results


def predict_logisticRegression(training_data, labels, test_data, labels_test):
    b_lr = LogisticRegression()
    b_lr.fit(training_data, labels)
    results = b_lr.predict(test_data)
    print("Logistic regression")
    probabilities = b_lr.predict_proba(test_data)
    with open("../results/logistic_results.txt", "wb") as fp:  # Pickling
        pickle.dump(results, fp)
    with open("../results/logistic_probabilities.txt", "wb") as fp:  # Pickling
        pickle.dump(results, fp)
    print(accuracy_score(labels_test, results))
    return results


def get_data():
    # data = pd.read_pickle('../data/data_features_clean.txt')
    # data = data[['rev_id', 'label', 'words_x', 'embedding']]
    # data = data.set_index('rev_id')
    data = pd.read_csv('../features/data_features_clean_flat.csv', header=None)
    return data


def get_features(data):

    features = data.drop([0, 1, 2, 3], axis=1)
    features = features.reset_index(drop=True)

    return features.values

print('Getting data...')
# data_0 = get_data()
# data = data_0.iloc[0:100]
# del data_0

data = get_data()

print('Getting features...')
features = get_features(data)

# print(data)

# features = data.drop(['label', 'words_x'], axis = 1)
# features = get_features(features)
# print(type(features))
# feature_names = ['embedding']
# features = data['embedding'].tolist()
print('Getting labels...')
labels = data[1].tolist()
print(labels)
# print(labels)
# print(type(labels[0]))
# count_labels_negative = len([x for x in labels if x == 0.0])
# print('Negative labels: ', count_labels_negative)

train, test, labels_train, labels_test = train_test_split(
    features, labels,
    test_size=0.20, random_state=42)

print('Starting classification...')
print(datetime.datetime.now())
print(predict_randomForest(train, labels_train, test, labels_test))
print('Finished Random Forest:', datetime.datetime.now())
print(predict_logisticRegression(train, labels_train, test, labels_test))
print('Finished Logistic Regression:', datetime.datetime.now())
