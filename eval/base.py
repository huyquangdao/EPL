import copy
import os
import copy
import re

from tqdm import tqdm
from dyna_gym.envs.utils import simulate_conversation, update_state, get_user_resp, get_llm_based_assessment
from dataset.data_utils import save_generated_conversations, construct_new_experience, save_new_experience
from collections import defaultdict

import time


class BaseOnlineEval(object):

    def __init__(self, target_set, terminal_act, horizon, use_llm_score=False, epsilon=1.0, n=5,
                 use_demonstration=False, k=3, dataset='durecdial'):
        self.terminal_act = terminal_act
        self.target_set = target_set
        self.horizon = horizon
        self.use_demonstration = use_demonstration
        self.use_llm_score = use_llm_score
        self.epsilon = epsilon
        self.n = n
        self.k = k
        self.dataset = dataset
        self.sr_turns = defaultdict(int)

        # initialize the value for sr@k
        # the inital values are zeros.
        for k in range(1, 2 * self.horizon + 1, 2):
            self.sr_turns[k] = 0

        # domain specific success rate and avg turns.

    def pipeline(self, state):
        raise NotImplementedError()

    def init_agent(self):
        raise NotImplementedError()

    def get_user_resp(self, state, system_resp):
        """
        method that simulates the user response
        @param state: the current state of the conversation
        @param system_resp: the generated system response
        @return: the generated user response
        """
        return get_user_resp(copy.deepcopy(state), system_resp, dataset=self.dataset)

    def init_state(self, target_item, system_initial_resp="Hello ! How do I help you ?"):
        """
        method that create the initial state of a conversation
        we assume the user start a conversation.
        @param target_item:
        @param system_initial_resp: The initial response from the system
        @return:
        """
        if 'conv' in target_item['demonstration']:
            del target_item['demonstration']['conv']

        if self.dataset == 'durecdial':
            goal = "Greetings"
            topic = "Greetings"
        else:
            goal = "no_strategy"
            topic = "no_strategy"

        state = {
            "task_background": {
                "target_topic": target_item['topic'],
                "target_goal": target_item['goal']
            },
            "demonstration": target_item["demonstration"],
            "dialogue_context": [],
            # "goal": "Greetings",  # will not affect anything, only including it for code convenience
            # "topic": "Greetings",
            "goal": goal,  # will not affect anything, only including it for code convenience
            "topic": topic,
            "knowledge": "",  # will not affect anything, only including it for code convenience
            "response": "",  # will not affect anything, only including it for code convenience
            "pre_goals": [],
            "pre_topics": []
        }
        user_initial_response = get_user_resp(state, sys_response=system_initial_resp, dataset=self.dataset)
        # state['dialogue_context'].append(
        #     {'role': 'system', 'content': system_initial_resp, 'act': (goal, topic)})
        state['dialogue_context'].append({'role': 'user', 'content': user_initial_response})
        return state

    def update(self, state, system_response, system_action, user_response):
        """
        method that update the state of the conversation
        @param state: the current state of the conversation
        @param system_response: the generated system response
        @param system_action: the predicted system action
        @param user_response: the generated user response
        @return:
        """
        new_state = update_state(state,
                                 action=system_action,
                                 sys_response=system_response,
                                 user_response=user_response
                                 )
        return new_state

    def run(self, init_state):
        """
        method that employs one online evaluation
        @param init_state:
        @return: a generated conversation between user and system
        """
        is_terminated = False
        count = 0
        generated_conversation = []


        # if init_state['task_background']['target_goal'] == 'Movie recommendation':
        #     print(init_state)

        state = init_state
        print("[TARGET]: ", state['task_background']['target_topic'])
        while not is_terminated and count < self.horizon:

            # generate system response and action
            system_resp, system_act = self.pipeline(state)

            # generate user response
            user_resp = self.get_user_resp(state, system_resp)

            # check the terminated condition
            if self.check_terminated_condition(system_act):
                is_terminated = True

            # if init_state['task_background']['target_goal'] == 'Movie recommendation':
            #     print('[SYSTEM ACT]: ', system_act)
            #     print('[SYSTEM]: ', system_resp)
            #     print('[USER]: ', user_resp)

            # update the state of the conversation
            state = self.update(state, system_resp, system_act, user_resp)

            # update count
            count += 1

            # update the simulated conversation
            generated_conversation.extend([
                {'role': 'system', 'content': system_resp, "act": system_act},
                {'role': 'user', 'content': user_resp}
            ])


        # if init_state['task_background']['target_goal'] == 'Movie recommendation':
        #     assert 1==0


        return generated_conversation

    def eval(self, saved_file_path=None, save_experience_path=None):
        """
        method that perform online evaluation on a predefined set of items
        @return: computed metrics
        """
        avg_srk = []
        avg_sr = []
        avg_turn = []
        all_generated_convs = []
        all_scores = []
        all_targets = []

        goal_specific_sr_dict = defaultdict(list)

        # objective metrics
        all_o_sr = []
        all_o_turns = []

        for target_item in tqdm(self.target_set):

            s_time = time.time()
            initial_state = self.init_state(target_item)
            generated_conversation = self.run(initial_state)
            print("Computational Time: ", time.time() - s_time)

            # LLM-based success rate.
            srk, sr, turn, score = self.compute_metrics(copy.deepcopy(generated_conversation), target_item['topic'],
                                                        initial_state[
                                                            'demonstration'] if self.use_demonstration else None)

            # Objective success rate.
            _, o_sr, o_turn = self.is_successful(
                generated_conversation=generated_conversation,
                target_item=target_item['topic']
            )

            # domain specific success rate and conversation turns
            # we save the subjective sr, avg turns and objective sr w.r.t different target goals.
            goal_specific_sr_dict[target_item['goal']].append((sr, turn, o_sr))

            generated_conversation = initial_state['dialogue_context'] + generated_conversation
            all_generated_convs.append(generated_conversation)
            avg_sr.append(sr)
            avg_turn.append(turn)
            avg_srk.append(srk)
            all_scores.append(score)
            all_targets.append(target_item['topic'])

            # objective metrics
            all_o_sr.append(o_sr)
            all_o_turns.append(o_turn)

        # saving generated conversations to file
        if saved_file_path is not None:
            save_generated_conversations(all_generated_convs, all_targets, saved_file_path)

        # constructing and saving the new experience
        if save_experience_path is not None:
            new_experience = construct_new_experience(all_generated_convs, all_scores)
            # save the new experience to the memory
            save_new_experience(new_experience, save_experience_path)

        # compute subjective sr_turns
        for k, v in self.sr_turns.items():
            self.sr_turns[k] = float(v) / len(self.target_set)

        # compute metrics w.r.t different domains.
        goal_metrics = defaultdict(dict)
        for k, v in goal_specific_sr_dict.items():
            # extreme case
            if len(v) == 0:
                goal_metrics[k]['sr'] = 0.0
                goal_metrics[k]['turn'] = self.horizon * 2
                goal_metrics[k]['o_sr'] = 0.0
                continue

            goal_metrics[k]['sr'] = 0.0
            goal_metrics[k]['avg_turn'] = 0.0
            goal_metrics[k]['o_sr'] = 0.0

            # compute the success rate, avg turn and objective sr w.r.t diffent target goal
            for sr, turn, o_sr in v:
                goal_metrics[k]['sr'] += int(sr)
                goal_metrics[k]['avg_turn'] += turn
                goal_metrics[k]['o_sr'] += int(o_sr)

            # taking the average
            goal_metrics[k]['sr'] = goal_metrics[k]['sr'] / len(v)
            goal_metrics[k]['avg_turn'] = goal_metrics[k]['avg_turn'] / len(v)
            goal_metrics[k]['o_sr'] = goal_metrics[k]['o_sr'] / len(v)

        # return success rate metrics and averaged conversation turns.
        return sum(avg_srk) / len(self.target_set), sum(avg_sr) / len(self.target_set), sum(avg_turn) / len(
            self.target_set), self.sr_turns, sum(all_o_sr) / len(all_o_sr), sum(all_o_turns) / len(
            all_o_turns), goal_metrics

    def check_terminated_condition(self, system_action):
        """
        method that check if the conversation is terminated
        @param system_action: the predicted system action
        @return: True if the conversaiton is terminated else False
        """
        return system_action == self.terminal_act

    def compute_metrics(self, generated_conversation, target_item, demonstrations=None):
        """
        method that compute the dialogue-level SR and avg number of conversational turn
        @param generated_conversation: set of generated conversations between user and system
        @param target_item: set of target item
        @return: dialogue-level SR and averaged number of conversational turn
        """
        score = None
        if self.use_llm_score:
            # compute success rate based on LLMs
            srk, sr, turn, llm_score = self.is_llm_based_successful(generated_conversation, target_item, demonstrations)
            score = llm_score
        else:
            # compute objective success rate.
            srk, sr, turn = self.is_successful(generated_conversation, target_item)
            score = int(sr)
        return int(srk), int(sr), turn, score

    def is_successful(self, generated_conversation, target_item):
        """
        method that check if the system successfully recommended the target item to the user.
        @param generated_conversation: the generated conversation between user and system
        @param target_item: the targeted item
        @return: True if success else False
        """
        # for inspired dataset
        if self.dataset == 'inspired':
            target_item = re.sub(r'\(\d+\)', '', target_item)

        for idx, utt in enumerate(generated_conversation):
            if utt['role'] == 'system' and target_item.lower().strip() in utt['content'].lower().strip():
                # successful before the k-th turn
                if idx + 1 <= self.k:
                    return True, True, idx + 1
                # the other case.
                else:
                    return False, True, idx + 1
        return False, False, len(generated_conversation)

    def is_llm_based_successful(self, generated_conversation, target_item, demonstrations):
        """
        method that return a score which is a LLM-based assessment
        @param generated_conversation: the generated conversation
        @param target_item: the target item
        @return: a float score
        """
        score = get_llm_based_assessment(target_item, copy.deepcopy(generated_conversation), demonstrations, n=self.n)

        # failed case.
        if score < self.epsilon:
            return False, False, len(generated_conversation), score

        # for inspired dataset
        if self.dataset == 'inspired':
            target_item = re.sub(r'\(\d+\)', '', target_item)

        # compute srk for every k
        # k = [0,1,.....,2 * horizon - 1]
        check = defaultdict(int)
        for k in range(1, 2 * self.horizon + 1, 2):
            # for utt in generated conversation
            for idx, utt in enumerate(generated_conversation):
                if utt['role'] == 'system' and target_item.lower().strip() in utt['content'].lower():
                    # successful before the k-th turn
                    # do not cound twice
                    if idx + 1 <= k and check[k] != 1:
                        self.sr_turns[k] += 1
                        check[k] = 1

        # identify the turn which the target is achieved.
        for idx, utt in enumerate(generated_conversation):
            if utt['role'] == 'system' and target_item.lower().strip() in utt['content'].lower():
                # successful before the k-th turn
                if idx + 1 <= self.k:
                    return True, True, idx + 1, score
                # the other case.
                else:
                    return False, True, idx + 1, score

        # failed to recommend the target item
        return False, False, len(generated_conversation), score

    def compute_turn(self, generated_conversation):
        """
        method that compute the number of turn needed to end the conversation
        @param generated_conversation: the generated conversation between user and system
        @return: a int number which stands for the number of conversational turn
        """
        return len(generated_conversation)
