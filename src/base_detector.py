import math
import numpy as np


class BaseDetector:
    def __init__(
        self,
        init_p: float,
        seed: int = 0,
        flag_additive_chain: bool = True,
        flag_iso_partition: bool = True,
    ):
        """Base detector/ Additive chain for AD2S

        Args:
            init_p (float): Init window reset probability.
            seed (int, optional): Random seed. Defaults to 0.
            flag_additive_chain (bool, optional): Enable flag for additive chains, for ablation study. Defaults to True.
            flag_iso_partition (bool, optional): Enable flag for isolation partition, for ablation study. Defaults to True.
        """

        self.window = {}
        self.begin_index = -1
        self.window_len = 0
        self.index = 0
        self.prob = init_p
        self.prob_alpha = None
        self.rng = np.random.default_rng(seed)
        self.sparse_cnt = 1

        # For Ablation experiment
        self.flag_additive_chain = flag_additive_chain
        self.flag_iso_partition = flag_iso_partition
        self.lst_nonzero_idx = -1

    def predict(self, data: float, index: int) -> float:
        """Predict the score of one observation

        Args:
            data (float): One observation.
            index (int): The index of the observation.

        Returns:
            float: Anomaly score for the current observation.
        """

        score = 0
        self.index = index
        self.window_len = self.index - self.begin_index
        if data >= 1:
            self.window[index] = data
            score = self.get_score(data)
            self.sparse_cnt = math.ceil(self.sparse_cnt / 2)
            self.update_prob() if self.flag_additive_chain else None
            self.lst_nonzero_idx = index

        if self.index == self.begin_index + 1:
            reset = 0
        else:
            if self.flag_additive_chain:
                reset = self.rng.binomial(1, self.prob, 1)[0]
            else:
                reset = (self.index + 1) % 10

        if reset == 1:
            self.window = {}
            self.sparse_cnt += 1
            self.update_prob() if self.flag_additive_chain else None
            self.begin_index = self.index

        return score

    def update_prob(self):
        """Update the window reset probability"""

        nonzero_cnt = len(self.window)

        new_prob = 1 / ((self.sparse_cnt + nonzero_cnt))

        self.prob_alpha = math.sqrt(self.prob) / (
            math.sqrt(self.prob) + self.prob
        )

        self.prob = (
            self.prob_alpha * new_prob + (1 - self.prob_alpha) * self.prob
        )

    def get_score(self, data: float) -> float:
        """Score the current observation with isolation partition

        Args:
            data (float): One observation.

        Returns:
            float: Anomaly score for the current observation.
        """

        if self.flag_iso_partition:
            level = 1
            nonzero_idx = list(self.window.keys())
            while len(nonzero_idx) > 1:
                spilt_idx = self.rng.integers(nonzero_idx[0], nonzero_idx[-1])
                nonzero_idx = [idx for idx in nonzero_idx if idx > spilt_idx]
                level += 1

            score = (level + math.log(data, 2)) / (
                math.log(self.index - self.begin_index, 2) + 1
            )

        else:

            score = len(self.window)

        return score
