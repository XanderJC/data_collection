import numpy as np
import copy


class BaseController:
    def __init__(self, n_actions=10, n_explainers=5):
        self.n_actions = n_actions
        self.n_explainers = n_explainers
        self.memory = []
        self.state = None

    def update(
        self,
        init_action,
        updated_action,
        explanations_viewed,
        example_index,
        explainers_given,
    ):

        self.memory.append(
            (
                copy.deepcopy(init_action),
                copy.deepcopy(updated_action),
                copy.deepcopy(explanations_viewed),
                copy.deepcopy(example_index),
                copy.deepcopy(explainers_given),
                copy.deepcopy(self.state),
            )
        )

        return

    def select_explainers(self):
        explainers = []
        return explainers
