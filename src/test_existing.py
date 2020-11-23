import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data_manager import DataManager

comments = DataManager.get_original_comments()
annotations = DataManager.get_annotations()

# labels a comment as an atack if the majority of annoatators did so
labels = annotations.groupby('rev_id')['attack'].mean() > 0.5

# join labels and comments# join
comments['attack'] = labels

# remove newline and tab tokens# remov
comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

comments.query('attack')['comment'].head()

ngrams = DataManager.get_ngrams()
new_labels = ngrams.merge(pd.DataFrame(comments['attack']), how='left', left_index=True, right_index=True)
features = ngrams.drop('rev_id', axis=1)

train_comments, test_comments, labels_train, labels_test = train_test_split(
    features, new_labels['attack'],
    test_size=0.20, random_state=42)

clf = LogisticRegression()
clf = clf.fit(train_comments, labels_train)
results = clf.predict(test_comments)
probabilities = clf.predict_proba(test_comments)

auc = roc_auc_score(labels_test, results)
print('Test ROC AUC: %.3f' %auc)
auc = roc_auc_score(labels_test, probabilities[:, 1])
print('Test ROC AUC: %.3f' %auc)