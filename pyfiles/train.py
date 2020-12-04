from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
import numpy as np
from joblibspark import register_spark
from sklearn.utils import parallel_backend
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import xgboost as xgb
import time


class SklearnTrainer(object):
    # helper class for sklearn models
    def __init__(self, classifier, seed = 0, params = None):
        if params:
            params['random_state'] = seed
            self.clf = classifier(**params)
        else:
            self.clf = classifier(random_state=seed)

        if 'probability' in self.clf.get_params():
            self.clf.set_params(probability=True)

    def fit(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def predict_prob(self, x):
        return self.clf.predict_proba(x)

    def feature_importance(self):
        return self.clf.feature_importances_


# create a dict of classifiers
model_dict = {"RandomForest":RandomForestClassifier, "LR":LogisticRegression, "DT":DecisionTreeClassifier, "SVM":SVC,
              "AdaBT":AdaBoostClassifier,"GDBT":GradientBoostingClassifier, "XGB":xgb.XGBClassifier}

# create a dict of models' parameters for grid/research search, you can define your own here.
param_dict = {"RandomForest":{}, "LR":{}, "DT":{}, "SVM":{}, "AdaBT":{}, "GDBT":{}, "XGB":{}}

# tuned hyper-parameters after randomized_search
tuned_params = {"RandomForest":{'max_depth': 80, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 1500, 'random_state': 42},
               "GBDT":{'learning_rate': 0.05, 'max_depth': 17, 'max_features': 'sqrt', 'min_samples_leaf': 8, 'min_samples_split': 5, 'n_estimators': 136, 'random_state': 42, 'subsample': 0.6},
                'XGB':{'colsample_bytree': 0.5, 'gamma': 0.05, 'learning_rate': 0.05, 'max_depth': 20, 'min_child_weight': 0.1, 'random_state': 42, 'reg_alpha': 1, 'reg_lambda': 0.5, 'subsample': 1}}

def dimension_reduction(X, n_comp):
    # using pca to reduce dimension of embedded text matrices, and return the desired feature matrix X
    pipeline = Pipeline([('scaler', StandardScaler()),('pca', PCA(n_components=n_comp))])
    tags_mat = pipeline.fit_transform(np.stack(X["tags_embedded"],axis=0))
    title_mat = pipeline.fit_transform(np.stack(X["title_embedded"], axis=0))
    dsp_mat = pipeline.fit_transform(np.stack(X["description_embedded"], axis=0))
    embed_mat = np.concatenate((tags_mat,title_mat,dsp_mat), axis = 1)
    scaler = StandardScaler()
    a = scaler.fit_transform(np.array(X[["comments_disabled", "ratings_disabled","time_gap"]]))
    X_reduced = np.concatenate((a,embed_mat), axis=1)
    return X_reduced


def grid_tuning(classifier, params, x_train, y_train, score):
    # hyper-parameter tuning using grid search
    register_spark()

    clf = classifier()
    with parallel_backend('spark', n_jobs=-1):
        clf_tuned = GridSearchCV(clf, param_grid=params, scoring=score, cv=5).fit(x_train, y_train)

    return clf_tuned


def random_tuning(classifier, params, x_train, y_train, n_itr, score):
    # hyper-parameter tuning using random search

    register_spark()

    clf = classifier()
    with parallel_backend('spark', n_jobs=-1):
        clf_tuned = RandomizedSearchCV(clf, param_distributions=params, n_iter=n_itr, scoring=score, cv=5, random_state=42).fit(x_train, y_train)

    return clf_tuned


def evaluate(y_true, y_score, config='ovr'):
    # roc score on test data
    return roc_auc_score(y_true, y_score, average='weighted', multi_class=config)


def baseline(clf_name, clf_dict, X, Y):
    # train on default parameter on given classifier and evaluate on two metrics
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
    clf_base = SklearnTrainer(clf_dict[clf_name])
    clf_base.fit(X_train, Y_train)
    roc_score_ovo = evaluate(Y_test, clf_base.predict_prob(X_test), 'ovo')
    roc_score_ovr = evaluate(Y_test, clf_base.predict_prob(X_test), 'ovr')


    return roc_score_ovo, roc_score_ovr


def simple_log_reg(X, Y, score='roc_auc_ovr_weighted'):
    # training on logistics regression
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8, random_state=42)
    grid_param = {"C":np.logspace(-3,3,7), "penalty":["l2","l1"], "multi_class":["multinomial"], "solver":["saga"], "max_iter":[500]}
    print("grid:", grid_param)
    lr = grid_tuning(LogisticRegression, grid_param, X_train, Y_train, score)
    best_param = lr.best_params_
    print("tuned parameter of LogisticRegression: ", best_param)
    return best_param


def simple_dt(X, Y, score='roc_auc_ovr_weighted'):
    # training on logistics regression
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8, random_state=42)
    grid_param = {"criterion":["gini","entropy"], "max_depth":[5,10,20,None], "min_samples_split":[2,5,8], "min_samples_leaf":[1,3,5]}
    print("grid:", grid_param)
    dt = grid_tuning(DecisionTreeClassifier, grid_param, X_train, Y_train, score)
    best_param = dt.best_params_
    print("tuned parameter of LogisticRegression: ", best_param)
    return best_param


def generic_train(clf_name, clf_dict, params_dict, X, Y, score='roc_auc_ovr_weighted', grid=True, n_itr=None):
    # given classifier name, dictionary of classifiers, their parameters and grid indicate whether use grid-search
    # tuning the hyper-parameters and fit the training set, get the best estimator and its parameter
    # with the test score on evaluation metric
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
    grid_param = params_dict[clf_name]
    clf = clf_dict[clf_name]
    if grid:
        clf_tuned = grid_tuning(clf, grid_param, X_train, Y_train, score)
    else:
        clf_tuned = random_tuning(clf, grid_param, X_train, Y_train, n_itr, score)
    best_param = clf_tuned.best_params_
    print(f"Tuned Parameter of {clf_name}", best_param)

    best_clf = SklearnTrainer(clf, params=best_param)
    start = time.time()
    best_clf.fit(X_train, Y_train)
    end = time.time()
    train_time = end-start
    test_score = evaluate(Y_test, best_clf.predict_prob(X_test))
    print(f"AUC score of {clf_name} on test set: {test_score}")
    print(f"Training time of {clf_name}: {train_time}s")

    return clf_name, best_clf, best_param, test_score, train_time,


def train(clf_dict, para_dict, X, Y, score='roc_auc_ovr_weighted', grid=True, n_itr=None):
    # get a dict of evaluated scores and best parameters for each classifier after hyper-tuning all the classifiers
    scores = {}
    params = {}
    time_cost = {}
    estimators = {}
    for clf_name in clf_dict:
        name, best_clf, best_param, eval_score, train_time = generic_train(clf_name, clf_dict, para_dict, X, Y, score, grid, n_itr)
        scores[name] = eval_score
        params[name] = best_param
        time_cost[name] = train_time
        estimators[name] = best_clf

    return scores, params, time_cost, estimators
