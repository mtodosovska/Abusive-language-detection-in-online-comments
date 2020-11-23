import pandas as pd
import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from data_manager import DataManager


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


    labels = DataManager.get_labels_2_classes().iloc[0:n, 1]
    # features = pd.read_csv(base_path + 'labels_to_test.csv').iloc[0:n, 2:3]
    embeddings = DataManager.get_embeddings().iloc[0:n, :]
    n_grams = DataManager.get_ngrams().iloc[0:n]
    scores = DataManager.get_scores().iloc[0:n, 2:]
    pos_tags = DataManager.get_pos_ngrams().iloc[0:n]
    linguistic = DataManager.get_linguistic().iloc[0:n]

    print('Merging features...')

    features = embeddings.merge(linguistic, on='0')
    features = n_grams.merge(features, left_on='rev_id', right_on='0').drop('0', axis=1)
    features = features.merge(scores, on='rev_id')
    features = features.merge(pos_tags, on='rev_id')


    return features.values, labels


print('Getting features and labels...')
features, labels = get_features()
print('Reducing features...')

print('Splitting training and test sets...')
train, test, labels_train, labels_test = train_test_split(
    features, labels,
    test_size=0.20, random_state=42)

classifiers = {'Logistic Regression': LogisticRegression(),
               'Random Forest': RandomForestClassifier(n_estimators=1000, n_jobs=-1, verbose=0, class_weight='balanced', bootstrap=True),
               'Naive Bayes': GaussianNB(),
               'LDA': LinearDiscriminantAnalysis(),
               'Decision Tree': DecisionTreeClassifier(random_state=0),
               'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=0),
               'SVM': SVC()
               }


for classifier in classifiers:
    print('Starting classification...')
    predict(classifier, classifiers[classifier], train, labels_train, test, labels_test)
    print('Finished ' + classifier, datetime.datetime.now())
