import numpy as np
from sklearn.naive_bayes import GaussianNB
from abc import ABC
class NBC(ABC):
    def __init__(self) -> None:
        self.cnt_zero = 0
        self.data_interval = []
        self.labels = []
        self.clf = GaussianNB()

    def fit(self, x, y):
        self.clf.fit(np.array(x).reshape(-1, 1), y)
        return self

    def score(self, x):
        label = self.clf.predict(np.array([[x]]))
        if label[0] == True:
            return 1
        else:
            return 0
