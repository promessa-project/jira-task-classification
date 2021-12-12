import os
import numpy as np
import pandas as pd
import features.build_features as feature

from sklearn.linear_model import SGDClassifier #SVM: SGDClassifier()
from sklearn.naive_bayes import MultinomialNB #MultinomialNB:: MultinomialNB()
from sklearn.ensemble import RandomForestClassifier #RandomForest:: RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=0)
from sklearn.svm import LinearSVC #SVM: LinearSVC()
from xgboost import XGBClassifier #XGBoost: XGBClassifier()

from sklearn.model_selection import train_test_split

import joblib

import models.predict_model as predict
from sklearn.metrics import accuracy_score

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

def train_model():
    print(os.path.join(os.getcwd(), 'artefacts/data/processed/jira_data.csv'))
    df = pd.read_csv(os.path.join(
        os.getcwd(), 'artefacts/data/processed/jira_data.csv'))

    # factorize the category (epic) for labels to improve the performance of classifiers
    df_encode = feature.encode_data(df)

    # iterate each model and train classifier and store in (models) folder
    for key in [1, 2, 3, 4, 5]:
        print('start training model using classifier: %s' % model[key])
        try:
            classifier = model[key]
            model_name = type(classifier).__name__

            x = df_encode['clean_summary'].values.astype('U')
            y = df_encode['epic']

            x = np.array(x)
            y = np.array(y)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = np.random.RandomState(0))
            count_vectorizer_output = feature.get_count_vectorizer_matrix(x_train)

            x_train_counts = count_vectorizer_output[0]
            # count_vect = count_vectorizer_output[1]

            joblib.dump(x_train_counts, "././models/feature/CountVectorizer.joblib")

            # tfidf matrix
            x_train_tfidf = feature.get_tfidf_matrix(x_train_counts)

            # joblib.dump(x_train_tfidf, "././models/TfidfTransformer.joblib")

            clf = classifier.fit(x_train_tfidf, y_train)

            print('model trained:: %s' % model[key])

            # predicted = predict.predict_model_cv(clf, count_vect, x_test)
            # print('accuracy:', accuracy_score(y_test, predicted))

            #Saving the machine learning model to a file for reuse while prediction
            joblib.dump(clf, "././models/"+model_name+".joblib")
        except Exception as e:
            print("Oops!", e.__class__, "occurred.")
            print("Skipping current model. Executing next model.")
            print()
