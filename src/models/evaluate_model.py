
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.pipeline import Pipeline

import numpy as np

def get_cross_val_task(df, proj, model):
    # get the train and test dataset for the projects
    project_data = df.loc[df['project_key'] == proj]
    model_scores = []

    X = project_data['clean_summary'].values.astype('U')
    y = project_data['epic']

    X = np.array(X)
    y = np.array(y)
    kf = KFold(n_splits=6, shuffle=True, random_state=1)

    accuracy_scores = []
    weighted_accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        count_vectorizer_output = get_count_vectorizer_matrix(X_train)
        X_train_counts = count_vectorizer_output[0]
        count_vect = count_vectorizer_output[1]

        # tfidf matrix
        X_train_tfidf = get_tfidf_matrix(X_train_counts)

        predicted = predict(model, count_vect, X_train_tfidf, y_train, X_test)

        weighted_score = (len(X_test) / len(X)) * accuracy_score(y_test, predicted)

        accuracy_scores.append(accuracy_score(y_test, predicted))
        weighted_accuracy_scores.append(weighted_score)
        f1_scores.append(f1_score(y_test, predicted, average='weighted'))
        precision_scores.append(precision_score(y_test, predicted, average='weighted'))
        recall_scores.append(recall_score(y_test, predicted, average='weighted'))

    return accuracy_scores, f1_scores, precision_scores, recall_scores, np.array(weighted_accuracy_scores).sum()


from sklearn.linear_model import SGDClassifier #SVM: SGDClassifier()
from sklearn.naive_bayes import MultinomialNB #MultinomialNB:: MultinomialNB()
from sklearn.ensemble import RandomForestClassifier #RandomForest:: RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=0)
from sklearn.svm import LinearSVC #SVM: LinearSVC()
from xgboost import XGBClassifier #XGBoost: XGBClassifier()
import pandas as pd

project_list = [
    'QUAEWATSICA',
    'ETLANG',
    'VELITONDRICKA',
    'MOLLITIAWEHNER',
    'EXLUEILWITZ',
    'ODIODURGAN-BOEHM',
    'REICIENDISMACGYVER',
    'OPTIOHERMANN-RUTHERFORD',
    'ODIORUECKER-WATSICA',
    'NISIRUTHERFORD-TROMP',
    'CORPORISVEUM-HEATHCOTE',
    'CONSEQUATURCASSIN-GLOVER',
#     'FACILISLUBOWITZ',
#     'FACILISJAKUBOWSKI',
]


accuracy_scores = []
f1_scores = []
precision_scores = []
recall_scores = []
weighted_accuracy_scores = []
results = []

model_accuracy_scores = []
model_f1_scores = []
model_precision_scores = []
model_recall_scores = []
model_weighted_accuracy_scores = []

model = {
    1: MultinomialNB(),
    2: RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=0),
    3: LinearSVC(),
    4: SGDClassifier(),
    5: XGBClassifier(num_class=6)
}

for key in [1, 2, 3, 4, 5]:
    print('model: %s' % model[key])
    for proj in project_list:
        print('project: %s' % proj)
        scores = get_cross_val_task(df_encode, proj, model[key])
        print('')
        accuracy_scores.append(np.array(scores[0]).mean())
        f1_scores.append(np.array(scores[1]).mean())
        precision_scores.append(np.array(scores[2]).mean())
        recall_scores.append(np.array(scores[3]).mean())
        weighted_accuracy_scores.append(np.array(scores[4]).mean())
    print('mean model accuracy score for %s => %s' % (proj, accuracy_scores))
    print('mean model f1 score for %s => %s' % (proj, f1_scores))
    print('mean model precision score for %s => %s' % (proj, precision_scores))
    print('mean model recall score for %s => %s' % (proj, recall_scores))
    print('mean model weighted accuracy score for %s => %s' % (proj, weighted_accuracy_scores))

    model_accuracy_scores.append(np.array(accuracy_scores).mean())
    model_f1_scores.append(np.array(f1_scores).mean())
    model_precision_scores.append(np.array(precision_scores).mean())
    model_recall_scores.append(np.array(recall_scores).mean())
    model_weighted_accuracy_scores.append(np.array(weighted_accuracy_scores).mean())

    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    weighted_accuracy_scores = []

result_data = {
    'Accuracy':model_accuracy_scores,
    'F1-Score':model_f1_scores,
    'Precision':model_precision_scores,
    'Recall':model_recall_scores,
    'Weighted accuracy':model_weighted_accuracy_scores
}
#
results = pd.DataFrame(result_data, index = ['Multinomial NB', 'Random Forest', 'LinearSVC', 'SGDClassifier', 'XGBoost'])
# results = pd.DataFrame(result_data, index = ['XGBoost'])
print('results:', results)
