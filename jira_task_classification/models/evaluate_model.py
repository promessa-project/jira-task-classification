
import os
import numpy as np
import pandas as pd
import features.build_features as feature

from sklearn.linear_model import SGDClassifier #SVM: SGDClassifier()
from sklearn.naive_bayes import MultinomialNB #MultinomialNB:: MultinomialNB()
from sklearn.ensemble import RandomForestClassifier #RandomForest:: RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=0)
from sklearn.svm import LinearSVC #SVM: LinearSVC()
from xgboost import XGBClassifier #XGBoost: XGBClassifier()

from sklearn.model_selection import KFold

import models.predict_model as predict

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

# models will be trained based on the data from below projects
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

# models to be trained
model = {
    1: MultinomialNB(),
    2: RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=0),
    3: LinearSVC(),
    4: SGDClassifier(),
    5: XGBClassifier(num_class=6)
}

accuracy_scores = []
weighted_accuracy_scores = []
f1_scores = []
precision_scores = []
recall_scores = []

model_accuracy_scores = []
model_f1_scores = []
model_precision_scores = []
model_recall_scores = []
model_weighted_accuracy_scores = []

def evaluate_model():
    print(os.path.join(os.getcwd(), 'artefacts/data/processed/jira_data.csv'))
    df = pd.read_csv(os.path.join(
        os.getcwd(), 'artefacts/data/processed/jira_data.csv'))

    # factorize the category (epic) for labels to improve the performance of classifiers
    df_encode = feature.encode_data(df)

    # iterate each model and train classifier and store in (models) folder
    for key in [1, 2, 3, 4, 5]:
        proj_accuracy_scores = []
        proj_weighted_accuracy_scores = []
        proj_f1_scores = []
        proj_precision_scores = []
        proj_recall_scores = []
        print('start training model using classifier: %s' % model[key])
        try:
            classifier = model[key]
            model_name = type(classifier).__name__
            for proj in project_list:
                accuracy_scores = []
                f1_scores = []
                precision_scores = []
                recall_scores = []
                weighted_accuracy_scores = []

                project_data = df_encode.loc[df_encode['project_key'] == proj]

                x = project_data['clean_summary'].values.astype('U')
                y = project_data['epic']

                x = np.array(x)
                y = np.array(y)

                # apply cross validation with 6-fold on project dataset
                kf = KFold(n_splits=6, shuffle=True, random_state=1)

                print('model evaluating for %s splits on project %s' % (kf.get_n_splits(x, y), proj))

                for train_index, test_index in kf.split(x, y):
                    x_train, x_test = x[train_index], x[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    count_vectorizer_output = feature.get_count_vectorizer_matrix(x_train)

                    x_train_counts = count_vectorizer_output[0]
                    count_vect = count_vectorizer_output[1]

                    # tfidf matrix
                    x_train_tfidf = feature.get_tfidf_matrix(x_train_counts)

                    clf = classifier.fit(x_train_tfidf, y_train)

                    predicted = predict.predict_model_cv(clf, count_vect, x_test)

                    accuracy_scores.append(accuracy_score(y_test, predicted))
                    weighted_score = (len(x_test) / len(x)) * accuracy_score(y_test, predicted)
                    weighted_accuracy_scores.append(weighted_score)
                    f1_scores.append(f1_score(y_test, predicted, average='weighted'))
                    precision_scores.append(precision_score(y_test, predicted, average='weighted'))
                    recall_scores.append(recall_score(y_test, predicted, average='weighted'))

                proj_accuracy_scores.append(np.array(accuracy_scores).mean())
                proj_f1_scores.append(np.array(f1_scores).mean())
                proj_precision_scores.append(np.array(precision_scores).mean())
                proj_recall_scores.append(np.array(recall_scores).mean())
                proj_weighted_accuracy_scores.append(np.array(weighted_accuracy_scores).sum())

            model_accuracy_scores.append(np.array(proj_accuracy_scores).mean())
            model_f1_scores.append(np.array(proj_f1_scores).mean())
            model_precision_scores.append(np.array(proj_precision_scores).mean())
            model_recall_scores.append(np.array(proj_recall_scores).mean())
            model_weighted_accuracy_scores.append(np.array(proj_weighted_accuracy_scores).mean())

            print('model evaluation completed: %s' % model_name)
        except Exception as e:
            print("Oops!", str(e) ,"occurred.")
            print("Skipping current model. Executing next model.")
            print()

    result_data = {
        'Accuracy':model_accuracy_scores,
        'F1-Score':model_f1_scores,
        'Precision':model_precision_scores,
        'Recall':model_recall_scores,
        'Weighted accuracy':model_weighted_accuracy_scores
    }

    results = pd.DataFrame(result_data, index = ['Multinomial NB', 'Random Forest', 'LinearSVC', 'SGDClassifier', 'XGBoost'])

    print('results:', results)
