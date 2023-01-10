from hla.controllers import BaseController
import random


class RandomController(BaseController):
    def __init__(self, n_actions=10, n_explainers=5):
        super().__init__(n_actions, n_explainers)

        self.explainers = list(range(n_explainers))

    def select_explainers(self):
        # n = random.randint(1, self.n_explainers)
        random.shuffle(self.explainers)
        return self.explainers
