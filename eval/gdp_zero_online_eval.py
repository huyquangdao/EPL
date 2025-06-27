import os
import pickle
import copy
from tqdm import tqdm

from dyna_gym.envs.utils import generate_knowledge_with_plm, \
    generate_sys_response_with_plm
import numpy as np
from eval.base import BaseOnlineEval
from baselines.gdp_zero.player import RTCPlayer, BERTPlayer
from baselines.gdp_zero.env import DialogGame
from baselines.gdp_zero.openloop_mcts import OpenLoopMCTS


class GDPZeroOnlineEval(BaseOnlineEval):

    def __init__(self, target_set, terminal_act, use_llm_score, n, k, rollouts, epsilon, use_demonstration
                 , generation_model, generation_tokenizer, know_generation_model,
                 know_generation_tokenizer,
                 policy_model, policy_tokenizer, horizon, goal2id, args=None,
                 device=None,
                 max_sequence_length=512, offline_policy=False, pad_to_multiple_of=True, padding='max_length',
                 max_gen_length=50, model_generation_args=None, should_plot_tree=True, use_rtcp_policy=True,
                 use_llama2=False, dataset='durecdial', topic2id=None
                 ):
        """
        constructor for class GDPZeroOnlineEval
        @param target_set:
        @param generation_model:
        @param generation_tokenizer:
        @param know_generation_model:
        @param know_generation_tokenizer:
        @param policy_model:
        @param policy_tokenizer:
        @param horizon:
        @param goal2id:
        @param device:
        @param max_sequence_length:
        @param offline_policy:
        @param max_gen_length:
        @param model_generation_args:
        @param should_plot_tree:
        """

        super().__init__(target_set, terminal_act, horizon, use_llm_score, epsilon, n, use_demonstration, k, dataset)
        self.generation_model = generation_model
        self.generation_tokenizer = generation_tokenizer
        self.know_generation_model = know_generation_model
        self.know_generation_tokenizer = know_generation_tokenizer
        self.policy_model = policy_model
        self.policy_tokenizer = policy_tokenizer
        self.horizon = horizon
        self.goal2id = goal2id
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        self.offline_policy = offline_policy
        self.max_gen_length = max_gen_length
        self.model_generation_args = model_generation_args
        self.should_plot_tree = should_plot_tree
        self.use_rtcp_policy = use_rtcp_policy
        self.topic2id = topic2id
        self.use_llama2 = use_llama2
        self.args = args
        self.rollouts = rollouts

        self.mcts_agent = self.init_agent()

    def init_agent(self):
        """
        method that initializes the Open-loop Monte-Carlo Tree Search agent
        @return: the constructed state.
        """
        # open-loop MCTS with RTCP prior policy
        if self.use_rtcp_policy:
            player = RTCPlayer(
                dataset=self.dataset,
                policy_model=self.policy_model,
                policy_tokenizer=self.policy_tokenizer,
                goal2id=self.goal2id,
                max_sequence_length=self.max_sequence_length,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                use_demonstration=self.use_demonstration,
                epsilon=self.epsilon,
                device=self.device,
                n=self.n,
            )
        else:
            # use BERT policy
            player = BERTPlayer(
                dataset=self.dataset,
                policy_model=self.policy_model,
                policy_tokenizer=self.policy_tokenizer,
                goal2id=self.goal2id,
                max_sequence_length=self.max_sequence_length,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                use_demonstration=self.use_demonstration,
                epsilon=self.epsilon,
                device=self.device,
                n=self.n,
            )
        game = DialogGame(
            dataset=self.dataset,
            generation_model=self.generation_model,
            generation_tokenizer=self.generation_tokenizer,
            know_generation_model=self.know_generation_model,
            know_generation_tokenizer=self.know_generation_tokenizer,
            goal2id=self.goal2id,
            horizon=self.horizon,
            max_sequence_length=self.max_sequence_length,
            max_gen_lenth=self.max_gen_length,
            device=self.device,
            n=self.n,
            use_demonstration=self.use_demonstration
        )

        self.mcts_agent = OpenLoopMCTS(game, player, self.args)
        return self.mcts_agent

    def update(self, state, system_response, system_action, user_response):
        # update state
        new_state = copy.deepcopy(state)
        new_state['dialogue_context'].append(
            {"role": "assistant", "content": system_response}
        )
        new_state['dialogue_context'].append(
            {"role": "user", "content": user_response}
        )
        goal, topic = system_action
        new_state['pre_goals'].append(goal)
        new_state['pre_topics'].append(topic)
        return new_state

    def check_terminated_condition(self, system_action):
        """
        function that check if the conversation is terminated
        @param system_action: the input system action (goal)
        @return: True if the conversation is terminated else False
        """
        return system_action[0] == self.terminal_act

    def pipeline(self, state):
        """
        method that perform one system pipeline including action prediction, knowledge generation and response generation
        @param state: the current state of the conversation
        @return: generated system response and predicted system action
        """

        # re-init the mcts algorithm at each step
        self.init_agent()

        # predict system action using open-loop monte-carlo tree search
        for _ in tqdm(range(self.rollouts)):
            self.mcts_agent.search(state)

        mcts_policy = self.mcts_agent.get_action_prob(state)
        action = self.mcts_agent.player.dialog_acts[np.argmax(mcts_policy)]

        if not self.use_llama2:
            # text generation with BART
            # generate relevant knowledge
            knowledge = generate_knowledge_with_plm(generation_model=self.know_generation_model,
                                                    tokenizer=self.know_generation_tokenizer,
                                                    action=action,
                                                    state=state,
                                                    max_sequence_length=self.max_sequence_length,
                                                    max_gen_length=self.max_gen_length,
                                                    pad_to_multiple_of=self.pad_to_multiple_of,
                                                    padding=self.padding,
                                                    device=self.device)

            # generate the system response using chatgpt
            # later it will be replaced by the generated response by BART.
            # system_resp = get_user_resp(start_state, action)
            system_resp = generate_sys_response_with_plm(generation_model=self.generation_model,
                                                         tokenizer=self.generation_tokenizer,
                                                         action=action,
                                                         knowledge=knowledge,
                                                         state=state,
                                                         max_sequence_length=self.max_sequence_length,
                                                         max_gen_length=self.max_gen_length,
                                                         pad_to_multiple_of=self.pad_to_multiple_of,
                                                         padding=self.padding,
                                                         device=self.device,
                                                         dataset=self.dataset
                                                         )

        else:
            # generating response with llama2
            # please implement a corresponding response generation function for llama 2
            system_resp = ''

        return system_resp, action
