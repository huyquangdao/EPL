from dyna_gym.default_policy.default_policy import DefaultPolicy

import gym
import torch
import numpy as np
from transformers import PreTrainedModel


class DialoguePlanningPolicy(DefaultPolicy):
    """
    Default policy that uses a HuggingFace transformer model.
    """
    def __init__(
            self,
            env: gym.Env,
            action_space: list,
            horizon: int,
    ):
        super().__init__(env, horizon)
        self.action_space = action_space

    @torch.no_grad()
    def get_predicted_sequence(self, state, horizon=None):
        return 1

    @torch.no_grad()
    def get_top_k_tokens(self, state):
        rand_prob = np.random.randn(len(self.action_space))
        top_1_index = np.argmax(rand_prob)
        return top_1_index, rand_prob[top_1_index]
