
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import optuna

def get_cross_val_task_optuna(trial):
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
#         'FACILISLUBOWITZ',
#         'FACILISJAKUBOWSKI',
    ]

    f1_scores = []

    for proj in project_list:
        f1_scores_split = []
        print('project: %s' % proj)
        df = df_encode
        project_data = df.loc[df['project_key'] == proj]

        print('project_data.shape::', project_data.shape)

        X = project_data['clean_summary']
        y = project_data['epic']

        X = np.array(X)
        y = np.array(y)
        kf = KFold(n_splits=6, shuffle=True, random_state=1)

# #         Model 1: Random Forest
#         n_estimators = trial.suggest_int('n_estimators', 2, 20)
#         max_depth = int(trial.suggest_loguniform('max_depth', 1, 32))
#         model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)


# #         Model 2: LinearSVC
#         model = SVC(kernel='linear', gamma="auto")

# #         Model 3: XGBoost
        param = {
                    "n_estimators" : trial.suggest_int('n_estimators', 0, 1000),
                    'max_depth':trial.suggest_int('max_depth', 2, 25),
                    'reg_alpha':trial.suggest_int('reg_alpha', 0, 5),
                    'reg_lambda':trial.suggest_int('reg_lambda', 0, 5),
                    'min_child_weight':trial.suggest_int('min_child_weight', 0, 5),
                    'gamma':trial.suggest_int('gamma', 0, 5),
                    'learning_rate':trial.suggest_loguniform('learning_rate',0.005,0.5),
                    'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.1,1,0.01),
                    'nthread' : -1
                }
        model = XGBClassifier(**param)


# #         Model 4: SVC
#          svc_c = trial.suggest_loguniform('svc_c', 1e-10, 1e10)
#          model = sklearn.svm.SVC(C=svc_c, gamma='auto')


        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            count_vectorizer_output = get_count_vectorizer_matrix(X_train)
            X_train_counts = count_vectorizer_output[0]
            count_vect = count_vectorizer_output[1]

            # tfidf matrix
            X_train_tfidf = get_tfidf_matrix(X_train_counts)

            predicted = predict(model, count_vect, X_train_tfidf, y_train, X_test)
            f1_scores_split.append(f1_score(y_test, predicted, average='weighted'))

        f1_scores.append(np.array(f1_scores_split).mean())

    print('f1_scores:', np.array(f1_scores).mean())
    return np.array(f1_scores).mean()



study = optuna.create_study(direction='maximize')
study.optimize(get_cross_val_task_optuna, n_trials=100)

trial = study.best_trial

print('F1 Score: {}'.format(trial.value))