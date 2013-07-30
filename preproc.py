import numpy as np
from scipy import sparse
from scipy.misc import factorial as fact
from itertools import combinations

from itertools import count
from collections import defaultdict

def do_get_feature(X, num_class, levels, levels_len):
    import numpy as np2
    results = []
    for j in xrange(X.shape[0]):
        result = []
        for i in xrange(num_class):
            index_offset = sum(levels_len[:i])
            found = np2.where(levels[i] == X[j,i])[0]
            if found.shape[0] == 0:
                if levels_len[i]>0:
                    result.append(index_offset + levels_len[i] - 1)
            else:
                index_feature = int(found)
                result.append(index_offset + index_feature)
        results.append(result)

    return results

def do_get_levels(X, class_idx, rare_threshold):
    import numpy as np2
    nrows = float(X.shape[0])
    levels = np2.unique(X[:,class_idx])
    levels_freq = np2.zeros(levels.shape[0])
    for i, level in enumerate(levels):
        levels_freq[i] = np2.where(X[:,class_idx] == level)[0].shape[0]/nrows
    return (levels[np2.where(levels_freq > rare_threshold)], 
                np2.any(levels_freq <= rare_threshold))

def interactions(X, degree=2):
    """
    Return a numpy array made of interactions terms of degree 'degree' 
    Interactions terms are represented as tuples.
    """
    def num_combinations(classes, degree):
        return int(fact(classes) / (fact(classes-degree) * fact(degree)))

    m, n = X.shape
    ncomb  = num_combinations(n, degree)
    Xinter = np.zeros((m, ncomb), dtype=tuple)
    for i, inds in enumerate(combinations(range(X.shape[1]), degree)):
        for j, v in enumerate(X[:,inds]):
            Xinter[j, i] = tuple(X[j, inds])

    return Xinter

class LabelEncoder:
    """
    Encode labels with value between 0 and n_classes-1.
    Takes and returns a numpy array
    """
    def __init__(self):
        pass

    def fit_transform(self, X):
        """
        Fit label encoder and return encoded labels.
        """
        m, n = X.shape
        X_labeled = np.zeros((m,n), dtype=np.int32) 
        self.counter = [count() for i in range(n)]
        self.counting_map = [defaultdict(self.counter[i].next) for i in range(n)]

        for col in range(n):
            for (row, v) in enumerate(X[:,col]):
                X_labeled[row, col] = (self.counting_map[col])[v]

        return X_labeled

    def fit(self, X):
        """
        Fit label encoder
        """
        self.fit_transform(X)

    def transform(self, X):
        """
        Transform labels to normalized encoding
        """
        m, n = X.shape
        X_labeled = np.zeros((m,n), dtype=np.int32) 

        for col in range(n):
            for (row, v) in enumerate(X[:,col]):
                X_labeled[row, col] = (self.counting_map[col])[v]

        return X_labeled


class OneHotEncoder:
    """
    Encode categorical integer features using a one-hot aka one-of-K scheme.
    """
    def __init__(self, rare_threshold=0):
        self.levels = []
        self.levels_len = []
        self.num_bin_features = 0
        self.num_class_features = 0
        self.rare_threshold = rare_threshold
        self.lb_view = None

    def set_lb_view(self, lb_view):
        self.lb_view = lb_view

    def __get_levels(self, X, class_idx):
        levels = np.unique(X[:,class_idx])
        levels_count = np.zeros(levels.shape[0])
        for i, level in enumerate(levels):
            levels_count[i] = np.where(X[:,class_idx] == level)[0].shape[0]
        return (levels[np.where(levels_count > self.rare_threshold)],
                    np.any(levels_count <= self.rare_threshold))

    def fit(self, X):
        self.num_class_features = X.shape[1]
        if self.lb_view:
            tasks=[]
            levels_all = []
            for i in xrange(self.num_class_features):
                task = self.lb_view.apply(do_get_levels, X, i,
                                            self.rare_threshold)
                tasks.append(task)
            for task in tasks:
                self.lb_view.wait(task)
                levels_all.append(task.get())
        else:
            levels_all = [self.__get_levels(X, i)
                            for i in xrange(self.num_class_features)]
        self.levels = [level for (level, rare) in levels_all]

        self.levels_len = [level.shape[0] + rare if level.shape[0]>0 else 0
                                                for (level,rare) in levels_all]
        self.num_bin_features = sum(self.levels_len)

    def __get_features(self, x):
        idxes = []
        for i in xrange(self.num_class_features):
            index_offset = sum(self.levels_len[:i])
            found = np.where(self.levels[i] == x[i])[0]
            if found.shape[0] == 0:
                if self.levels_len[i]>0:
                    idxes.append(index_offset + self.levels_len[i] - 1)
            else:
                index_feature = int(found)
                idxes.append(index_offset + index_feature)
    
        return idxes
    
    def transform(self, X):
        n = X.shape[0]
        X_new = sparse.lil_matrix((n, self.num_bin_features), dtype=np.int32)
        if self.lb_view:
            tasks = []
            group_size = 1000
            num_group = n/group_size;

            for i in xrange(num_group):
                task = self.lb_view.apply(do_get_feature,
                                X[range(i*group_size, (i+1)*group_size),:],
                                self.num_class_features,
                                self.levels, self.levels_len)
                tasks.append(task)
            if n - num_group*group_size:
                task = self.lb_view.apply(do_get_feature,
                                X[range(num_group*group_size, n),:],
                                self.num_class_features,
                                self.levels, self.levels_len)
                tasks.append(task)

            for i, task in enumerate(tasks):
                self.lb_view.wait(task)
                idxes_list=task.get()
                for j,idxes in enumerate(idxes_list):
                    X_new[i*group_size+j, idxes] = 1
        else:
            idxes_list = do_get_feature(X,
                                    self.num_class_features,
                                    self.levels, self.levels_len)
            for i,idxes in enumerate(idxes_list):
                X_new[i, idxes] = 1

        return sparse.csc_matrix(X_new)

    # def transform(self, X):
    #     n = X.shape[0]
    #     X_new = sparse.lil_matrix((n, self.num_bin_features), dtype=np.int32)
    #     for i in xrange(n):
    #         idxes = self.__get_features(X[i,:])
    #         X_new[i, idxes] = 1
    #         #for idx in idxes:
    #         #    X_new[i, idx] = 1
        
    #     return sparse.csc_matrix(X_new)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def map_indices(self, inds):
        """
        Map old column indices to new column indices.
        """
        new_inds = []
        for ind in inds:
            offset = sum(self.levels_len[:ind])
            new_inds.extend(range(offset, offset + self.levels_len[ind]))

        return new_inds
            