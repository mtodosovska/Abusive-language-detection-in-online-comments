import pandas as pd
import datetime

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


class SimpleTransformerGeneral(TransformerMixin):

    def transform(self, X, **transform_params):
        # ls = X.drop('comment', axis=1).drop('words', axis=1)
        # print(ls.shape)
        return X


    def fit(self, X, y=None, **fit_params):
        return self


def get_balanced_labels():
    print('Getting labels')
    labels = pd.read_csv('../features/labels_2_classes.csv').iloc[:, :]
    labels_1 = labels[labels.label == 1]
    labels_0 = labels[labels.label == 0]
    # labels_0 = labels_0.sample(frac=1).iloc[0:labels_1.shape[0]]
    labels_0 = labels_0.iloc[0:labels_1.shape[0]]

    labels_0.to_csv('../features/labels_00.csv')
    labels_1.to_csv('../features/labels_11.csv')

    print('Got labels  ')
    labels_all = labels_0.append(labels_1)
    # print(labels_all)
    return labels_all, labels_1.shape[0]


def get_data(n):
    print('Getting data')
    print(datetime.datetime.now())
    labels, m = get_balanced_labels()
    # m = 2*m

    # scores = pd.read_csv('../features/scores.csv', encoding='latin1').drop('Unnamed: 0', axis=1).iloc[0:n]
    # gi = pd.read_csv('../features/harvard4.csv', encoding='latin1')
    # scores = scores.merge(gi, on='rev_id')
    # del gi

    # pos_tags = pd.read_csv('../features/pos_ngrams.csv', encoding='latin1').drop('Unnamed: 0', axis=1).drop('index', axis=1).iloc[0:n]
    # # scores = pos_tags
    # scores = scores.merge(pos_tags, on='rev_id')
    # del pos_tags
    #
    ngrams = pd.read_csv('../features/ngrams.csv', encoding='latin1').drop('Unnamed: 0', axis=1).iloc[0:m]
    scores = ngrams
    scores = scores.merge(ngrams, on='rev_id')
    del ngrams
    #
    offensiveness = pd.read_csv('../features/offensiveness_score.csv', encoding='latin1').iloc[0:m]
    scores = scores.merge(offensiveness, on='rev_id')
    del offensiveness

    embeddings = pd.read_csv('../features/embeddings.csv', encoding='latin1').iloc[0:m]
    # linguistic = pd.read_csv('../features/linguistic.csv', encoding='latin1').iloc[0:n]
    # embeddings = embeddings.merge(linguistic, on='0')
    # del linguistic

    # labels = pd.read_csv('../features/labels_2_classes.csv', encoding='latin1')

    new_labels = labels.merge(scores, how='left', left_on='rev_id', right_on='rev_id')

    # print(new_labels)
    # del labels

    new_labels = new_labels.merge(embeddings, left_on='rev_id', right_on='0').drop('0', axis=1)
    # new_labels = new_labels.merge(scores, left_on='rev_id', right_on='rev_id')

    del embeddings
    del labels
    del scores

    features = new_labels.drop('label', axis=1)#.drop('rev_id', axis=1)

    return features, new_labels


def evaluate(labels_test, results, probabilities):
    print(classification_report(labels_test, results, target_names=['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5']))
    print(confusion_matrix(labels_test, results))

    auc = roc_auc_score(labels_test, results)
    print('Test ROC AUC (predict): %.3f' % auc)
    auc = roc_auc_score(labels_test, probabilities[:, 1])
    print('Test ROC AUC (predict_proba): %.3f' % auc)
    print(datetime.datetime.now())
    print('---------------------------------------------------')

    print('Precision:', precision_score(labels_test, results))
    print('Recall:', recall_score(labels_test, results))
    print('F1:', f1_score(labels_test, results))



n = 115615
# n = 100
#
features, new_labels = get_data(n)

print('Splitting data')
train, test, labels_train, labels_test = train_test_split(
    features, new_labels['label'],
    test_size=0.20, random_state=42)

del features
del new_labels
# print('Fitting pipeline')
# clf = Pipeline([
#     ('features', SimpleTransformerGeneral()),
#     # ('LogisticRegression', LogisticRegression(class_weight='balanced')),
#     ('RandomForest', RandomForestClassifier(n_estimators=1000, n_jobs=-1, verbose=0, class_weight='balanced', bootstrap=True))
# ])

models = []
models.append(('LogisticRegression', LogisticRegression()))
# models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
# models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
# models.append(('GaussianNB', GaussianNB()))
# models.append(('RandomForestClassifier', RandomForestClassifier()))
# models.append(('ExtraTreesClassifier', ExtraTreesClassifier()))
# models.append(('SVM', SVC()))

for name, model in models:
    clf = model
    print(name)
    print('Fitting and classifying')
    print(datetime.datetime.now())
    clf = clf.fit(train.iloc[:, 1:], labels_train)
    results = clf.predict(test.iloc[:, 1:])
    probabilities = clf.predict_proba(test.iloc[:, 1:])
    # print(results)
    evaluate(labels_test, results, probabilities)

    fn = 0
    fp = 0
    tp = 0
    tn = 0

    l = 0

    for r, i, j, k in zip(test.iloc[:, 0], labels_test, results, probabilities):
        if l <= 50:
            print(r, i, j, k)
            l += 1
        if i == 1 and j == 0:
            fn += 1
        elif i == 1 and j == 1:
            tp += 1
        elif i == 0 and j == 1:
            fp += 1
        else:
            tn += 1

    print('TP', tp)
    print('FP', fp)
    print('TN', tn)
    print('FN', fn)




