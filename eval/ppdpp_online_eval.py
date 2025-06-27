import os
import pickle
import copy

from dyna_gym.envs.utils import generate_knowledge_with_plm, \
    generate_sys_response_with_plm
from eval.base import BaseOnlineEval


class PPDPPBartOnlineEval(BaseOnlineEval):
    """
    Online evaluation class for RTCP with Bart knowledge and text generation model
    """

    def __init__(self, target_set, terminal_act, use_llm_score, n, k, epsilon, use_demonstration, generation_model,
                 generation_tokenizer,
                 know_generation_model,
                 know_generation_tokenizer,
                 policy_model, policy_tokenizer, horizon, goal2id, device=None,
                 max_sequence_length=512, pad_to_multiple_of=True, padding='max_length',
                 max_gen_length=50, dataset='durecdial'
                 ):
        """
        constructor for class MCTSCRSOnlineEval
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
        @param max_gen_length:
        """

        super().__init__(target_set, terminal_act, horizon, use_llm_score, epsilon, n, use_demonstration, k,
                         dataset=dataset)
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
        self.max_gen_length = max_gen_length

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
        # predict action with PPDPP policy
        action = self.policy_model.select_action(state=state, is_test=True)

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
        return system_resp, action
