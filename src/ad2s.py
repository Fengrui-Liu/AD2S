from .base_detector import BaseDetector
import numpy as np
import time


class AD2SDetector:
    def __init__(
        self,
        n_base: int = 20,
        init_p: float = 0.1,
        seed: int = 0,
        flag_additive_chain: bool = True,
        flag_iso_partition: bool = True,
    ):
        """AD2S Detector

        Args:
            n_base (int, optional): The number of base additive chains. Defaults to 20.
            init_p (float, optional): The init probability of window reset. Defaults to 0.1.
            seed (int, optional): Random seed. Defaults to 0.
            flag_additive_chain (bool, optional): Enable flag for additive chains, for ablation study. Defaults to True.
            flag_iso_partition (bool, optional): Enable flag for isolation partition, for ablation study. Defaults to True.
        """

        self.n_base = n_base
        self.index = -1
        self.p = init_p
        self.window_size = 0

        self.base = []
        self.time_list = []
        self.score_list = []
        self.rng = np.random.default_rng(seed)

        for seed in self.rng.integers(0, n_base * 1000, size=n_base):
            self.base.append(
                BaseDetector(
                    init_p=self.p,
                    seed=seed,
                    flag_additive_chain=flag_additive_chain,
                    flag_iso_partition=flag_iso_partition,
                )
            )

    def predict(self, data: float) -> float:
        """Predict the score of one observation.

        Args:
            data (float): One observation.

        Returns:
            float: Anomaly score for the current observation.
        """

        self.index += 1
        score = 0

        # Used for following the states of the detector
        self.time_list = []
        self.score_list = []
        self.window_size = 0
        self.p = 0

        start_time = time.perf_counter()
        for idx, detector in enumerate(self.base):
            s = detector.predict(data, index=self.index)
            score += s
            self.score_list.append(s)
            self.window_size += detector.window_len
            self.p += detector.prob
            self.time_list.append(time.perf_counter() - start_time)

        self.window_size = self.window_size / self.n_base
        self.p = self.p / self.n_base

        p_score = score / self.n_base

        return p_score
