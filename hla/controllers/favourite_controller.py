from hla.controllers import BaseController


class FavouriteController(BaseController):
    def __init__(self, favourite, n_actions=10, n_explainers=5):
        super().__init__(n_actions, n_explainers)

        self.favourite = favourite
        self.explainers = list(range(n_explainers))

    def select_explainers(self):

        return [self.explainers[self.favourite]]
