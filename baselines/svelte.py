from abc import ABC


class SvelteDetector(ABC):
    def __init__(
        self,
        threshold=200,
    ):
        self.threshold = threshold
        self.index = -1
        self.last_index = -1

    def score(self, data):
        self.index += 1

        score = 0

        if data == 0:
            score = 0
        else:
            gap = self.index - self.last_index
            self.last_index = self.index
            if gap < self.threshold:
                score = 1
            else:
                score = 0

        return score
