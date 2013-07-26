import numpy as np
import copy
from sklearn import metrics
from sklearn.base import BaseEstimator
from IPython.parallel import interactive

@interactive
def do_fit(classifier, dump_file, get_features_func, features_index):
    from sklearn.externals import joblib
    X, y = joblib.load(dump_file, mmap_mode='r')
    #features_list = joblib.load(features_list_file, mmap_mode='r')
    features = get_features_func(features_index)
    return classifier.fit(X[:, features], y)

@interactive
def do_predict_proba(classifier, dump_file,
                    get_features_func, features_index):
    from sklearn.externals import joblib
    X = joblib.load(dump_file, mmap_mode='r')
    #features_list = joblib.load(features_list_file, mmap_mode='r')
    features = get_features_func(features_index)
    return classifier.predict_proba(X[:, features])

@interactive
def do_fit_predict(classifier, dump_file_train, dump_file_test,
                    get_features_func, features_index):
    from sklearn.externals import joblib
    Xtrain, ytrain = joblib.load(dump_file_train, mmap_mode='r')
    Xtest = joblib.load(dump_file_test, mmap_mode='r')
    features = get_features_func(features_index)

    classifier.fit(Xtrain[:, features], ytrain)
    return classifier.predict_proba(Xtest[:, features])


class Ensemble(BaseEstimator):
    """
    Take a list of classifiers and combine them.
    """
    def __init__(self, classifiers, weights):
        self.classifiers = classifiers
        self.weights = weights

    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X,y)
        return self

    def predict_proba(self, Xtest):
        y = np.zeros(Xtest.shape[0])
        for classifier, weight in zip(self.classifiers, self.weights):
            y += weight * classifier.predict_proba(Xtest)

    def score(self, X, y, method='auc'):
        yhat = self.predict_proba(X)
        fpr, tpr, thresholds = metrics.roc_curve(y, yhat, pos_label=1)
        return metrics.auc(fpr, tpr)


class EnsembleFeature(BaseEstimator):
    """
    Create a ensemble using a single type of classifier but with different
    feature selected.
    """
    def __init__(self, classifier, num_features, get_features_fun, weights,
                                dump_path='dumps/splits/'):
        self.classifiers = [copy.deepcopy(classifier)
                                for i in xrange(num_features)]
        self.get_features_fun = get_features_fun
        self.weights = weights
        self.dump_path = dump_path
        self.lb_view=None

    def set_lb_view(self, lb_view):
        self.lb_view = lb_view

    def fit(self, X, y):
        if self.lb_view:
            from sklearn.externals import joblib
            tasks = []
            dump_file = self.dump_path + 'ensemble_fit_Xyfeat.pkl'
            joblib.dump((X,y), dump_file)
            for i, classifier in enumerate(self.classifiers):
                task = self.lb_view.apply(do_fit, classifier, dump_file,
                                            self.get_features_fun, i)
                tasks.append(task)

            for i, task in enumerate(tasks):
                self.lb_view.wait(task)
                self.classifiers[i] = task.get()
        else:
            for i, classifier in enumerate(self.classifiers):
                features = self.get_features_fun(i)
                classifier.fit(X[:, features], y)


        return self

    def predict_proba(self, Xtest):
        y = np.zeros((Xtest.shape[0], 2))

        if self.lb_view:
            from sklearn.externals import joblib
            tasks = []
            dump_file = self.dump_path + 'ensemble_predict_X.pkl'
            joblib.dump(Xtest, dump_file)
            for i, classifier in enumerate(self.classifiers):
                task = self.lb_view.apply(do_predict_proba, classifier,
                                            dump_file, self.get_features_fun, i)
                tasks.append(task)

            for task, weight in zip(tasks, self.weights):
                self.lb_view.wait(task)
                y += weight * task.get()
        else:
            for i,classifier, features, weight in enumerate(self.classifiers):
                features = self.get_features_fun(i)
                y += self.weights[i] * classifier.predict_proba(Xtest[:, features])

        return y

    def fit_predict(self, Xtrain, ytrain, Xtest):
        y = np.zeros((Xtest.shape[0], 2))

        if self.lb_view:
            from sklearn.externals import joblib
            tasks = []
            dump_file_train = self.dump_path + 'ensemble_fit_Xyfeat.pkl'
            joblib.dump((Xtrain,ytrain), dump_file_train)
            dump_file_test = self.dump_path + 'ensemble_predict_X.pkl'
            joblib.dump(Xtest, dump_file_test)
            for i, classifier in enumerate(self.classifiers):
                task = self.lb_view.apply(do_fit_predict, classifier,
                                            dump_file_train, dump_file_test,
                                            self.get_features_fun, i)
                tasks.append(task)

            for task, weight in zip(tasks, self.weights):
                self.lb_view.wait(task)
                y += weight * task.get()
        else:
            for i,classifier, features, weight in enumerate(self.classifiers):
                features = self.get_features_fun(i)
                classifier.fit(Xtrain[:, features], ytrain)
                y += self.weights[i] * classifier.predict_proba(Xtest[:, features])

        return y

    def score(self, X, y, method='auc'):
        yhat = self.predict_proba(X)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(y, yhat, pos_label=1)
        return metrics.auc(fpr, tpr)

    def fit_score(self, Xtrain, ytrain, Xtest, ytest, method='auc'):
        ytest_hat = self.fit_predict(Xtrain, ytrain, Xtest)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(ytest, ytest_hat, pos_label=1)
        return metrics.auc(fpr, tpr)
