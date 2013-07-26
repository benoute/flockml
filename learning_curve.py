import numpy as np
from math import log10
import pylab as pl
import os


def compute_score(model, data_file, index, train_size, permutation):
    from sklearn.externals import joblib
    (X, y) = joblib.load(data_file, mmap_mode='r')

    Xtrain = X[permutation[:train_size], :]
    ytrain = y[permutation[:train_size], :]
    Xtest  = X[permutation[train_size:], :]
    ytest  = y[permutation[train_size:], :]

    model.fit(Xtrain, ytrain)
    score_train = model.score(Xtrain, ytrain)
    score_test  = model.score(Xtest,  ytest)
    return (index, score_train, score_test)


class LearningCurve:
    def __init__(self, lb_view, model, X, y, resolution=20, cv_folds=5,
                    dump_folder='dumps/splits'):
        from sklearn.externals import joblib
        self.resolution = resolution
        self.lb_view = lb_view

        set_size         = X.shape[0]
        test_size        = set_size/cv_folds
        max_train_size   = set_size - test_size
        self.train_sizes = np.logspace(2, log10(max_train_size), self.resolution).astype(np.int)
        permutations     = [np.random.permutation(set_size) for i in xrange(cv_folds)]

        self.tasks  = []

        data_file = os.path.join(dump_folder, "learningcurve.pkl")
        joblib.dump((X, y), data_file)

        for i, train_size in enumerate(self.train_sizes):
            for permutation in permutations:
                task = self.lb_view.apply(compute_score, model, data_file,
                                            i, train_size, permutation)
                self.tasks.append(task)


    def progress(self):
        ready = sum([1 for task in self.tasks if task.ready()])
        print "%i%%" % (100*ready / len(self.tasks))

    def wait(self):
        for task in self.tasks:
            self.lb_view.wait(task)

    def get_scores(self):
        scores_train = [[] for i in xrange(self.resolution)]
        scores_test  = [[] for i in xrange(self.resolution)]

        for task in self.tasks:
            if not task.ready(): 
                continue

            index, score_train, score_test = task.get()
            scores_train[index].append(score_train)
            scores_test[index].append(score_test)

        return (self.train_sizes, [np.mean(score) for score in scores_train], 
                             [np.mean(score) for score in scores_test])

    def plot(self):
        sizes, scores_train, scores_test = self.get_scores()
        pl.plot(sizes, scores_train, 'o-k', c='g', label='Train Score')
        pl.plot(sizes, scores_test, 'o-k', c='b', label='Test Score')
        pl.xlabel("Training set size")
        pl.ylabel("Score")
        pl.legend(loc='best')
        pl.title("Learning Curve")


