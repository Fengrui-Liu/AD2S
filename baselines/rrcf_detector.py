from collections import deque

import numpy as np
import rrcf
from .base_detector import BaseDetector
from copy import deepcopy


class RrcfDetector(BaseDetector):
    def __init__(self, num_trees=20, tree_size=64, seed=0, **kwargs):
        """Rrcf detector :cite:`DBLP:conf/icml/GuhaMRS16`.

        Args:
            window_len (int, optional): Length of sliding window. Defaults to 150.
            num_trees (int, optional): Number of trees. Defaults to 20.
            tree_size (int, optional): Size of each tree. Defaults to 64.
        """

        super().__init__(data_type="univariate", **kwargs)
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.forest = []
        self.rng = np.random.default_rng(seed)
        for seed in self.rng.integers(0, num_trees*100, num_trees):
            tree = rrcf.RCTree(random_state=seed)
            self.forest.append(tree)
        self.avg_codisp = {}

        self.shingle = deque(maxlen=int(np.sqrt(self.window_len)))
        self.shingle.extend([0] * int(np.sqrt(self.window_len)))

    def fit(self, X: np.ndarray, timestamp: int = None):

        self.shingle.append(X[0])

        for tree in self.forest:
            if len(tree.leaves) > self.tree_size:
                tree.forget_point(self.index - self.tree_size)

            tree.insert_point(self.shingle, self.index)

        return self

    def score(self, X: np.ndarray, timestamp: int = None):

        score_list = []
        shingle = deepcopy(self.shingle)
        shingle.pop()
        shingle.append(X[0])
        for tree in self.forest:
            try:
                query_idx = tree.query(X[0])
                score_list.append(tree.codisp(query_idx))
            except:
                tree.insert_point(shingle, "tmp")
                score_list.append(tree.codisp("tmp"))
                tree.forget_point("tmp")

        score = np.mean(score_list)

        return float(score)
