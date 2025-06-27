import os
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence

from dataset.base import BaseTorchDataset
from config.config import IGNORE_INDEX


def max_seq_length(list_l):
    return max(len(l) for l in list_l)


def pad_sequence(list_l, max_len, padding_value=0):
    assert len(list_l) <= max_len
    padding_l = [padding_value] * (max_len - len(list_l))
    padded_list = list_l + padding_l
    return padded_list


class UnimindTorchDataset(BaseTorchDataset):

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess the input instances for unimind model
        @param instances: a set of input instances.
        @param convert_example_to_feature: a dictionary which contains key and values where values are functions
        @return: a set of processed input instances.
        """
        if isinstance(convert_example_to_feature, dict):
            processed_instances = []
            # loop overall instances.
            for instance in instances:
                # loop overall functions.
                for key, func in convert_example_to_feature.items():
                    input_ids, label = func(self.tokenizer, instance, self.max_sequence_length, self.max_target_length,
                                            is_test=self.is_test, is_gen=self.is_gen)

                    new_instance = {
                        "input_ids": input_ids,
                        "label": label
                    }
                    processed_instances.append(new_instance)

            return processed_instances
        else:
            return super().preprocess_data(instances, convert_example_to_feature)


class GPTTorchDataset(BaseTorchDataset):

    def collate_fn(self, batch):
        """
        method that construct tensor-kind of inputs for DialogGPT, GPT2 models.
        @param batch: the input batch
        @return: a dictionary of torch tensors.
        """
        input_features = defaultdict(list)
        labels_gen = []
        context_length_batch = []
        for instance in batch:
            # construct the input for decoder-style of pretrained language model
            # the input is the concaternation of dialogue context and response
            # the label is similar to the input but we mask all position corresponding to the dialogue context
            # the label will be shifted to the right direction in the model
            input_features['input_ids'].append(instance['input_ids'])
            context_length_batch.append(len(instance['input_ids']))
            labels_gen.append(instance['label'])

        # if inference time, then input padding should be on the left hand side.
        if self.is_test:
            self.tokenizer.padding_size = 'left'
        # if training time, then input padding should be on the right hand side.
        else:
            self.tokenizer.padding_size = 'right'

        # padding the input features
        input_features = self.tokenizer.pad(
            input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # labels for response generation task, for computing the loss function
        labels = input_features['input_ids']
        labels = [[token_id if token_id != self.tokenizer.pad_token_id else IGNORE_INDEX for token_id in resp] for resp
                  in labels]

        labels = torch.as_tensor(labels, device=self.device)

        # labels for response generation task, for computing generation metrics.
        labels_gen = torch_pad_sequence(
            [torch.tensor(label, dtype=torch.long) for label in labels_gen],
            batch_first=True, padding_value=IGNORE_INDEX)
        labels_gen = labels_gen.to(self.device)

        # convert features to torch tensors
        for k, v in input_features.items():
            if not isinstance(v, torch.Tensor):
                input_features[k] = torch.as_tensor(v, device=self.device)

        new_batch = {
            "context": input_features,
            "labels": labels,
            "labels_gen": labels_gen,
            "context_len": context_length_batch
        }
        return new_batch


class RTCPTorchDataset(BaseTorchDataset):

    def __init__(self, tokenizer, instances, goal2id=None, topic2id=None, max_sequence_length=512, padding='max_length',
                 pad_to_multiple_of=True, device=None, convert_example_to_feature=None, max_target_length=50,
                 is_test=False, is_gen=False):
        """
        constructor for the BaseTorchDataset Class
        @param tokenizer: an huggingface tokenizer
        @param instances: a list of instances
        @param goal2id: a dictionary which maps goal to index.
        @param max_sequence_length: the maximum length of the input sequence.
        @param padding: type of padding
        @param pad_to_multiple_of: pad to multiple instances
        @param device: device to allocate the data, eg: cpu or gpu
        @param convert_example_to_feature: a function that convert raw instances to
        corresponding inputs and labels for the model.
        @param max_target_length the maximum number of the target sequence (response generation only)
        @param is_test True if inference step False if training step
        @param is_gen True if response generation else False
        @param is_inference True if testing time else False
        """
        self.topic2id = topic2id
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.goal2id = goal2id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        self.device = device
        self.max_target_length = max_target_length
        self.is_test = is_test
        self.is_gen = is_gen
        self.instances = self.preprocess_data(instances, convert_example_to_feature)

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess the input data for the RTCP model.
        @param instances: a list of input instances.
        @param convert_example_to_feature: a function that processes the given input instance.
        @return:
        """
        processed_instances = []
        for instance in instances:
            # data processing for training the policy model
            if not self.is_gen:
                context_ids, path_ids, label_goal, label_topic = convert_example_to_feature(self.tokenizer, instance,
                                                                                            self.max_sequence_length,
                                                                                            self.goal2id,
                                                                                            self.topic2id)
                new_instance = {
                    "context_ids": context_ids,
                    "path_ids": path_ids,
                    "label_goal": label_goal,
                    "label_topic": label_topic
                }
            # data processing for training the generation model
            else:
                input_ids, label, goal_id, topic_id = convert_example_to_feature(self.tokenizer, instance,
                                                                                 self.max_sequence_length,
                                                                                 self.max_target_length,
                                                                                 self.goal2id,
                                                                                 self.topic2id,
                                                                                 is_test=self.is_test)
                new_instance = {
                    "input_ids": input_ids,
                    "label": label,
                    "goal_id": goal_id,
                    "topic_id": topic_id
                }
            processed_instances.append(new_instance)
        return processed_instances

    def collate_fn(self, batch):
        """
        method that construct tensor-kind of inputs for DialogGPT, GPT2 models.
        @param batch: the input batch
        @return: a dictionary of torch tensors.
        """
        # collate function for training the policy model
        if not self.is_gen:
            context_input_features = defaultdict(list)
            path_input_features = defaultdict(list)
            labels_goal = []
            labels_topic = []
            for instance in batch:
                context_input_features['input_ids'].append(instance['context_ids'])
                path_input_features['input_ids'].append(instance['path_ids'])
                labels_goal.append(instance['label_goal'])
                labels_topic.append(instance['label_topic'])

            # padding the context features
            context_input_features = self.tokenizer.pad(
                context_input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
                max_length=self.max_sequence_length
            )
            # convert features to torch tensors
            for k, v in context_input_features.items():
                if not isinstance(v, torch.Tensor):
                    context_input_features[k] = torch.as_tensor(v, device=self.device)

            # padding the path features
            path_input_features = self.tokenizer.pad(
                path_input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
                max_length=self.max_sequence_length
            )
            # convert features to torch tensors
            for k, v in path_input_features.items():
                if not isinstance(v, torch.Tensor):
                    path_input_features[k] = torch.as_tensor(v, device=self.device)

            labels_goal = torch.LongTensor(labels_goal).to(self.device)
            labels_topic = torch.LongTensor(labels_topic).to(self.device)

            new_batch = {
                "context": context_input_features,
                "path": path_input_features,
                "labels_goal": labels_goal,
                "labels_topic": labels_topic
            }
        # collate function for response generation model
        else:
            input_features = defaultdict(list)
            labels_gen = []
            context_length_batch = []
            goal_indices = []
            topic_indices = []
            for instance in batch:
                # construct the input for decoder-style of pretrained language model
                # the input is the concaternation of dialogue context and response
                # the label is similar to the input but we mask all position corresponding to the dialogue context
                # the label will be shifted to the right direction in the model
                input_features['input_ids'].append(instance['input_ids'])
                context_length_batch.append(len(instance['input_ids']))
                labels_gen.append(instance['label'])
                goal_indices.append(instance['goal_id'])
                topic_indices.append(instance['topic_id'])

            # padding the input features
            # if inference time, then input padding should be on the left hand side.
            if self.is_test:
                self.tokenizer.padding_size = 'left'
            # if training time, then input padding should be on the right hand side.
            else:
                self.tokenizer.padding_size = 'right'

            input_features = self.tokenizer.pad(
                input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
                max_length=self.max_sequence_length
            )

            # labels for response generation task, for computing the loss function
            labels = input_features['input_ids']
            labels = [[token_id if token_id != self.tokenizer.pad_token_id else IGNORE_INDEX for token_id in resp] for
                      resp
                      in labels]
            labels = torch.as_tensor(labels, device=self.device)

            # labels for response generation task, for computing generation metrics.
            labels_gen = torch_pad_sequence(
                [torch.tensor(label, dtype=torch.long) for label in labels_gen],
                batch_first=True, padding_value=IGNORE_INDEX)
            labels_gen = labels_gen.to(self.device)

            # convert features to torch tensors
            for k, v in input_features.items():
                if not isinstance(v, torch.Tensor):
                    input_features[k] = torch.as_tensor(v, device=self.device)

            # create goal and topic tensors.
            goal_indices = torch.LongTensor(goal_indices).to(self.device)
            topic_indices = torch.LongTensor(topic_indices).to(self.device)

            input_features['goal_ids'] = goal_indices
            input_features['topic_ids'] = topic_indices

            new_batch = {
                "context": input_features,
                "labels": labels,
                "labels_gen": labels_gen,
                "context_len": context_length_batch,
            }
        return new_batch


def list_to_tensor(list_l, padding_idx=0, device=None):
    max_len = max_seq_length(list_l)
    padded_lists = []
    for list_seq in list_l:
        padded_lists.append(pad_sequence(list_seq, max_len, padding_value=padding_idx))
    input_tensor = torch.tensor(padded_lists, dtype=torch.long)
    input_tensor = input_tensor.to(device).contiguous()
    return input_tensor


def varlist_to_tensor(list_vl, padding_idx=0, device=None):
    lens = []
    for list_l in list_vl:
        lens.append(max_seq_length(list_l))
    max_len = max(lens)

    padded_lists = []
    for list_seqs in list_vl:
        v_list = []
        for list_l in list_seqs:
            v_list.append(pad_sequence(list_l, max_len, padding_value=padding_idx))
        padded_lists.append(v_list)
    input_tensor = torch.tensor(padded_lists, dtype=torch.long)
    input_tensor = input_tensor.to(device).contiguous()
    return input_tensor


def get_attention_mask(data_tensor: torch.tensor, padding_idx=0, device=None):
    attention_mask = data_tensor.masked_fill(data_tensor == padding_idx, 0)
    attention_mask = attention_mask.masked_fill(attention_mask != padding_idx, 1)
    attention_mask = attention_mask.to(device).contiguous()
    return attention_mask


class TCPTorchDataset(BaseTorchDataset):

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess the input data for the TCP model.
        @param instances: a list of input instances.
        @param convert_example_to_feature: a function that processes the given input instance.
        @return:
        """
        processed_instances = []
        for instance in instances:
            # data processing for training the policy model
            feature = convert_example_to_feature(self.tokenizer, instance,
                                                 self.max_sequence_length,
                                                 self.is_test
                                                 )
            processed_instances.append(feature)
        return processed_instances

    def collate_fn(self, mini_batch):
        """Custom collate function for dealing with batches of input data.
        Arguments:
            mini_batch: A list of input features.
        Return:
            dict: (dict) A dict of tensors.
        """
        up_ids = []
        kg_ids, kg_segs, kg_poss, kg_hops = [], [], [], []
        hs_ids, hs_segs, hs_poss = [], [], []
        tg_ids = []
        input_ids, gold_ids = [], []
        for sample in mini_batch:
            up_ids.append(sample.user_profile_ids)
            kg_ids.append(sample.knowledge_ids)
            kg_segs.append(sample.knowledge_segs)
            kg_poss.append(sample.knowledge_poss)
            kg_hops.append(sample.knowledge_hops)
            hs_ids.append(sample.conversation_ids)
            hs_segs.append(sample.conversation_segs)
            hs_poss.append(sample.conversation_poss)

            tg_ids.append(sample.target_ids)
            input_ids.append(sample.input_ids)
            gold_ids.append(sample.gold_ids)

        batch_up_ids = list_to_tensor(up_ids, device=self.device)
        batch_up_masks = get_attention_mask(batch_up_ids, device=self.device)

        batch_kg_ids = list_to_tensor(kg_ids, device=self.device)
        batch_kg_segs = list_to_tensor(kg_segs, device=self.device)
        batch_kg_poss = list_to_tensor(kg_poss, device=self.device)
        batch_kg_hops = list_to_tensor(kg_hops, device=self.device)
        batch_kg_masks = get_attention_mask(batch_kg_ids, device=self.device)

        batch_hs_ids = list_to_tensor(hs_ids, device=self.device)
        batch_hs_segs = list_to_tensor(hs_segs, device=self.device)
        batch_hs_poss = list_to_tensor(hs_poss, device=self.device)
        batch_hs_masks = get_attention_mask(batch_hs_ids, device=self.device)

        batch_tg_ids = list_to_tensor(tg_ids, device=self.device)
        batch_tg_masks = get_attention_mask(batch_tg_ids, device=self.device)

        batch_input_ids = list_to_tensor(input_ids, device=self.device)
        batch_input_masks = get_attention_mask(batch_input_ids, device=self.device)
        batch_gold_ids = list_to_tensor(gold_ids, device=self.device)

        collated_batch = {
            "user_profile": [batch_up_ids, batch_up_masks],
            "knowledge": [batch_kg_ids, batch_kg_segs, batch_kg_poss, batch_kg_hops, batch_kg_masks],
            "conversation": [batch_hs_ids, batch_hs_segs, batch_hs_poss, batch_hs_masks],
            "target": [batch_tg_ids, batch_tg_masks],
            "plan": [batch_input_ids, batch_input_masks, batch_gold_ids]
        }

        return collated_batch

    @staticmethod
    def static_collate_fn(mini_batch, device=None):
        up_ids = []
        kg_ids, kg_segs, kg_poss, kg_hops = [], [], [], []
        hs_ids, hs_segs, hs_poss = [], [], []
        tg_ids = []
        input_ids, gold_ids = [], []
        for sample in mini_batch:
            up_ids.append(sample.user_profile_ids)
            kg_ids.append(sample.knowledge_ids)
            kg_segs.append(sample.knowledge_segs)
            kg_poss.append(sample.knowledge_poss)
            kg_hops.append(sample.knowledge_hops)
            hs_ids.append(sample.conversation_ids)
            hs_segs.append(sample.conversation_segs)
            hs_poss.append(sample.conversation_poss)

            tg_ids.append(sample.target_ids)
            input_ids.append(sample.input_ids)
            gold_ids.append(sample.gold_ids)

        batch_up_ids = list_to_tensor(up_ids, device=device)
        batch_up_masks = get_attention_mask(batch_up_ids, device=device)

        batch_kg_ids = list_to_tensor(kg_ids, device=device)
        batch_kg_segs = list_to_tensor(kg_segs, device=device)
        batch_kg_poss = list_to_tensor(kg_poss, device=device)
        batch_kg_hops = list_to_tensor(kg_hops, device=device)
        batch_kg_masks = get_attention_mask(batch_kg_ids, device=device)

        batch_hs_ids = list_to_tensor(hs_ids, device=device)
        batch_hs_segs = list_to_tensor(hs_segs, device=device)
        batch_hs_poss = list_to_tensor(hs_poss, device=device)
        batch_hs_masks = get_attention_mask(batch_hs_ids, device=device)

        batch_tg_ids = list_to_tensor(tg_ids, device=device)
        batch_tg_masks = get_attention_mask(batch_tg_ids, device=device)

        batch_input_ids = list_to_tensor(input_ids, device=device)
        batch_input_masks = get_attention_mask(batch_input_ids, device=device)
        batch_gold_ids = list_to_tensor(gold_ids, device=device)

        collated_batch = {
            "user_profile": [batch_up_ids, batch_up_masks],
            "knowledge": [batch_kg_ids, batch_kg_segs, batch_kg_poss, batch_kg_hops, batch_kg_masks],
            "conversation": [batch_hs_ids, batch_hs_segs, batch_hs_poss, batch_hs_masks],
            "target": [batch_tg_ids, batch_tg_masks],
            "plan": [batch_input_ids, batch_input_masks, batch_gold_ids]
        }

        return collated_batch


def planner_list_to_tensor(list_l, special_padding_value=None, padding_idx=None, device=None):
    max_len = max_seq_length(list_l)
    padded_lists = []
    for list_seq in list_l:
        if special_padding_value is None:
            padded_lists.append(pad_sequence(list_seq, max_len, padding_value=padding_idx))
        else:
            padded_lists.append(pad_sequence(list_seq, max_len, padding_value=special_padding_value))
    input_tensor = torch.tensor(padded_lists, dtype=torch.long)
    input_tensor = input_tensor.to(device).contiguous()
    return input_tensor


class COLORBridgeTorchDataset(BaseTorchDataset):

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess the input data for the TCP model.
        @param instances: a list of input instances.
        @param convert_example_to_feature: a function that processes the given input instance.
        @return:
        """
        processed_instances = []
        for instance in instances:
            # data processing for training the policy model
            features = convert_example_to_feature(self.tokenizer, instance,
                                                  self.max_sequence_length,
                                                  self.is_test
                                                  )
            processed_instances.extend(features)
        return processed_instances

    def collate_fn(self, mini_batch):
        """Custom collate function for dealing with batches of input data.
        Arguments:
            mini_batch: A list of input features.
        Return:
            dict: (dict) A dict of tensors.
        """
        batch_user_utterance_input = []
        batch_delta_follow_input = []
        batch_interim_subgoal_input = []
        batch_start_subgoal_input = []
        batch_target_subgoal_input = []
        batch_transition_input = []
        batch_interim_t, batch_target_T = [], []

        for sample in mini_batch:
            batch_user_utterance_input.append(sample.user_utt_ids)
            batch_delta_follow_input.append(sample.follow_ids)
            batch_transition_input.append(sample.transition_ids)
            batch_interim_subgoal_input.append(sample.interim_ids)
            batch_start_subgoal_input.append(sample.start_ids)
            batch_target_subgoal_input.append(sample.target_ids)
            batch_interim_t.append(sample.interim_t)
            batch_target_T.append(sample.target_T)

        # inputs
        user_utt_ids = list_to_tensor(batch_user_utterance_input, device=self.device)
        user_utt_mask = get_attention_mask(user_utt_ids, device=self.device)

        delta_follow_ids = list_to_tensor(batch_delta_follow_input, device=self.device)
        delta_follow_mask = get_attention_mask(delta_follow_ids, device=self.device)

        transition_ids = list_to_tensor(batch_transition_input, device=self.device)
        transition_mask = get_attention_mask(transition_ids, device=self.device)

        interim_subgoal_ids = list_to_tensor(batch_interim_subgoal_input, device=self.device)
        interim_subgoal_mask = get_attention_mask(interim_subgoal_ids, device=self.device)

        start_subgoal_ids = list_to_tensor(batch_start_subgoal_input, device=self.device)
        start_subgoal_mask = get_attention_mask(start_subgoal_ids, device=self.device)

        target_subgoal_ids = list_to_tensor(batch_target_subgoal_input, device=self.device)
        target_subgoal_mask = get_attention_mask(target_subgoal_ids, device=self.device)

        interim_t = torch.tensor(batch_interim_t, dtype=torch.long).to(self.device).contiguous()
        target_T = torch.tensor(batch_target_T, dtype=torch.long).to(self.device).contiguous()

        collated_batch = {
            "user_utterance": [user_utt_ids, user_utt_mask],
            "delta_follow": [delta_follow_ids, delta_follow_mask],
            "transition": [transition_ids, transition_mask],
            "interim_subgoal": [interim_subgoal_ids, interim_subgoal_mask],
            "start_subgoal": [start_subgoal_ids, start_subgoal_mask],
            "target_subgoal": [target_subgoal_ids, target_subgoal_mask],
            "interim_t": interim_t,
            "target_T": target_T,
        }

        return collated_batch

    @staticmethod
    def static_collate_fn(mini_batch, device=None):
        pass


class COLORPlanningTorchDataset(BaseTorchDataset):

    def __init__(self, model, latent_dim, tokenizer, instances, goal2id=None, max_sequence_length=512,
                 padding='max_length',
                 pad_to_multiple_of=True, device=None, convert_example_to_feature=None, max_target_length=50,
                 is_test=False, is_gen=False):
        super().__init__(tokenizer, instances, goal2id, max_sequence_length, padding, pad_to_multiple_of, device,
                         convert_example_to_feature, max_target_length, is_test, is_gen)
        self.model = model
        self.latent_dim = latent_dim

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess the input data for the TCP model.
        @param instances: a list of input instances.
        @param convert_example_to_feature: a function that processes the given input instance.
        @return:
        """
        processed_instances = []
        m = -1
        for instance in instances:
            # data processing for training the policy model
            feature = convert_example_to_feature(self.tokenizer, instance,
                                                 self.max_sequence_length,
                                                 self.is_test
                                                 )
            m = max(feature.transition_number, m)
            processed_instances.append(feature)
        print(m)
        return processed_instances

    def collate_fn(self, mini_batch):
        """
        Collate function for the planner class
        """
        batch_input = []
        batch_decoder_input_all = []
        batch_transition_input = []
        batch_start_subgoal_input = []
        batch_target_subgoal_input = []
        batch_user_utterance_input = []
        batch_delta_follow_input = []
        batch_transition_number = []

        for sample in mini_batch:
            batch_input.append(sample.input_ids)
            batch_decoder_input_all.append(sample.decoder_input_all_ids)
            batch_transition_input.append(sample.transition_ids)
            batch_start_subgoal_input.append(sample.start_ids)
            batch_target_subgoal_input.append(sample.target_ids)
            batch_user_utterance_input.append(sample.user_utt_ids)
            batch_delta_follow_input.append(sample.follow_ids)
            batch_transition_number.append(sample.transition_number)

        input_ids = list_to_tensor(batch_input, device=self.device)
        input_mask = get_attention_mask(input_ids, device=self.device)
        decoder_input_all_ids = list_to_tensor(batch_decoder_input_all, device=self.device)
        decoder_input_all_mask = get_attention_mask(decoder_input_all_ids, device=self.device)

        start_subgoal_ids = list_to_tensor(batch_start_subgoal_input, device=self.device)
        start_subgoal_mask = get_attention_mask(start_subgoal_ids, device=self.device)

        target_subgoal_ids = list_to_tensor(batch_target_subgoal_input, device=self.device)
        target_subgoal_mask = get_attention_mask(target_subgoal_ids, device=self.device)

        user_utt_ids = list_to_tensor(batch_user_utterance_input, device=self.device)
        user_utt_mask = get_attention_mask(user_utt_ids, device=self.device)

        delta_follow_ids = list_to_tensor(batch_delta_follow_input, device=self.device)
        delta_follow_mask = get_attention_mask(delta_follow_ids, device=self.device)

        transition_number = torch.tensor(batch_transition_number, dtype=torch.long).to(self.device).contiguous()

        batch_tc_mask = []
        for bsz, sample in enumerate(mini_batch):
            tc_mask_temp = (len(sample.decoder_input_ids_list) - 1) * [1]
            batch_tc_mask.append(tc_mask_temp)
        tc_mask = planner_list_to_tensor(batch_tc_mask, special_padding_value=0, device=self.device)
        gold_temp = torch.full((tc_mask.shape[0], tc_mask.shape[1], self.latent_dim), 0, dtype=torch.float).to(
            self.device)
        for bsz, sample in enumerate(mini_batch):
            if len(sample.decoder_input_ids_list) > 1:
                for idx, dec_ids in enumerate(sample.decoder_input_ids_list):
                    if idx == len(sample.decoder_input_ids_list) - 1:
                        continue
                    temp_ids = list_to_tensor([dec_ids], device=self.device)
                    temp_mask = get_attention_mask(temp_ids, device=self.device)
                    gold_temp[bsz, idx, :] = self.model.get_time_control_embed(temp_ids, temp_mask)

        simulate_temp = torch.full((tc_mask.shape[0], tc_mask.shape[1], self.latent_dim), 0, dtype=torch.float).to(
            self.device)
        for bsz, sample in enumerate(mini_batch):
            if len(sample.decoder_input_ids_list) > 1:
                start_latent = self.model.get_time_control_embed(start_subgoal_ids[bsz:bsz + 1, :],
                                                                 start_subgoal_mask[bsz:bsz + 1, :])
                target_latent = self.model.get_time_control_embed(target_subgoal_ids[bsz:bsz + 1, :],
                                                                  target_subgoal_mask[bsz:bsz + 1, :])
                Z_u = self.model.get_user_utt_representation(user_utt_ids[bsz:bsz + 1, :],
                                                             user_utt_mask[bsz:bsz + 1, :])
                delta_u = self.model.get_delta_u_representation(delta_follow_ids[bsz:bsz + 1, :],
                                                                delta_follow_mask[bsz:bsz + 1, :])

                # simulate Brownian bridge trjectories
                simulate_bridge_points = self.model.simulate_brownian_bridge(B_0=start_latent, B_T=target_latent,
                                                                             T=len(sample.decoder_input_ids_list),
                                                                             Z_u=Z_u, delta_u=delta_u)

                assert len(simulate_bridge_points) == len(sample.decoder_input_ids_list)
                for idx, embed in enumerate(simulate_bridge_points[1:]):
                    simulate_temp[bsz, idx, :] = embed

        if not self.is_test:
            collated_batch = {
                "input": [input_ids, input_mask],
                "decoder_input_all": [decoder_input_all_ids[:, :-1].contiguous(),
                                      decoder_input_all_mask[:, :-1].contiguous()],
                "label": [decoder_input_all_ids[:, 1:].contiguous(), decoder_input_all_mask[:, 1:].contiguous()],
                "transition_number": transition_number,
                "gold_bridge_embed": [gold_temp.contiguous(), tc_mask],
                "simulate_bridge_embed": [simulate_temp.contiguous(), tc_mask],
            }
        else:
            gold_bridge_list = []
            for bsz, sample in enumerate(mini_batch):
                tc_list = []
                if len(sample.decoder_input_ids_list) > 1:
                    for idx, dec_ids in enumerate(sample.decoder_input_ids_list):
                        if idx == len(sample.decoder_input_ids_list) - 1:
                            continue
                        temp_ids = list_to_tensor([dec_ids], device=self.device)
                        temp_mask = get_attention_mask(temp_ids, device=self.device)
                        rep = self.model.get_time_control_embed(temp_ids, temp_mask)
                        tc_list.append(rep)
                gold_bridge_list.append(tc_list)
            collated_batch = {
                "input": [input_ids, input_mask],
                "decoder_input_all": [decoder_input_all_ids[:, :-1].contiguous(),
                                      decoder_input_all_mask[:, :-1].contiguous()],
                "transition_number": transition_number,
                "gold_bridge_embed": [gold_temp.contiguous(), tc_mask],
                "simulate_bridge_embed": [simulate_temp.contiguous(), tc_mask],
                "user_utterance": [user_utt_ids, user_utt_mask],
                "delta_follow": [delta_follow_ids, delta_follow_mask],
                "start_subgoal": [start_subgoal_ids, start_subgoal_mask],
                "target_subgoal": [target_subgoal_ids, target_subgoal_mask],
                "gold_bridge_list": gold_bridge_list,
            }

        return collated_batch

    @staticmethod
    def static_collate_fn(mini_batch, model, device=None, latent_dim=16, is_test=True):
        batch_input = []
        batch_decoder_input_all = []
        batch_transition_input = []
        batch_start_subgoal_input = []
        batch_target_subgoal_input = []
        batch_user_utterance_input = []
        batch_delta_follow_input = []
        batch_transition_number = []

        for sample in mini_batch:
            batch_input.append(sample.input_ids)
            batch_decoder_input_all.append(sample.decoder_input_all_ids)
            batch_transition_input.append(sample.transition_ids)
            batch_start_subgoal_input.append(sample.start_ids)
            batch_target_subgoal_input.append(sample.target_ids)
            batch_user_utterance_input.append(sample.user_utt_ids)
            batch_delta_follow_input.append(sample.follow_ids)
            batch_transition_number.append(sample.transition_number)

        input_ids = list_to_tensor(batch_input, device=device)
        input_mask = get_attention_mask(input_ids, device=device)
        decoder_input_all_ids = list_to_tensor(batch_decoder_input_all, device=device)
        decoder_input_all_mask = get_attention_mask(decoder_input_all_ids, device=device)

        start_subgoal_ids = list_to_tensor(batch_start_subgoal_input, device=device)
        start_subgoal_mask = get_attention_mask(start_subgoal_ids, device=device)

        target_subgoal_ids = list_to_tensor(batch_target_subgoal_input, device=device)
        target_subgoal_mask = get_attention_mask(target_subgoal_ids, device=device)

        user_utt_ids = list_to_tensor(batch_user_utterance_input, device=device)
        user_utt_mask = get_attention_mask(user_utt_ids, device=device)

        delta_follow_ids = list_to_tensor(batch_delta_follow_input, device=device)
        delta_follow_mask = get_attention_mask(delta_follow_ids, device=device)

        transition_number = torch.tensor(batch_transition_number, dtype=torch.long).to(device).contiguous()

        batch_tc_mask = []
        for bsz, sample in enumerate(mini_batch):
            tc_mask_temp = (len(sample.decoder_input_ids_list) - 1) * [1]
            batch_tc_mask.append(tc_mask_temp)
        tc_mask = planner_list_to_tensor(batch_tc_mask, special_padding_value=0, device=device)
        gold_temp = torch.full((tc_mask.shape[0], tc_mask.shape[1], latent_dim), 0, dtype=torch.float).to(
            device)
        for bsz, sample in enumerate(mini_batch):
            if len(sample.decoder_input_ids_list) > 1:
                for idx, dec_ids in enumerate(sample.decoder_input_ids_list):
                    if idx == len(sample.decoder_input_ids_list) - 1:
                        continue
                    temp_ids = list_to_tensor([dec_ids], device=device)
                    temp_mask = get_attention_mask(temp_ids, device=device)
                    gold_temp[bsz, idx, :] = model.get_time_control_embed(temp_ids, temp_mask)

        simulate_temp = torch.full((tc_mask.shape[0], tc_mask.shape[1], latent_dim), 0, dtype=torch.float).to(
            device)
        for bsz, sample in enumerate(mini_batch):
            if len(sample.decoder_input_ids_list) > 1:
                start_latent = model.get_time_control_embed(start_subgoal_ids[bsz:bsz + 1, :],
                                                            start_subgoal_mask[bsz:bsz + 1, :])
                target_latent = model.get_time_control_embed(target_subgoal_ids[bsz:bsz + 1, :],
                                                             target_subgoal_mask[bsz:bsz + 1, :])
                Z_u = model.get_user_utt_representation(user_utt_ids[bsz:bsz + 1, :],
                                                        user_utt_mask[bsz:bsz + 1, :])
                delta_u = model.get_delta_u_representation(delta_follow_ids[bsz:bsz + 1, :],
                                                           delta_follow_mask[bsz:bsz + 1, :])

                # simulate Brownian bridge trjectories
                simulate_bridge_points = model.simulate_brownian_bridge(B_0=start_latent, B_T=target_latent,
                                                                        T=len(sample.decoder_input_ids_list),
                                                                        Z_u=Z_u, delta_u=delta_u)

                assert len(simulate_bridge_points) == len(sample.decoder_input_ids_list)
                for idx, embed in enumerate(simulate_bridge_points[1:]):
                    simulate_temp[bsz, idx, :] = embed

        if not is_test:
            collated_batch = {
                "input": [input_ids, input_mask],
                "decoder_input_all": [decoder_input_all_ids[:, :-1].contiguous(),
                                      decoder_input_all_mask[:, :-1].contiguous()],
                "label": [decoder_input_all_ids[:, 1:].contiguous(), decoder_input_all_mask[:, 1:].contiguous()],
                "transition_number": transition_number,
                "gold_bridge_embed": [gold_temp.contiguous(), tc_mask],
                "simulate_bridge_embed": [simulate_temp.contiguous(), tc_mask],
            }
        else:
            gold_bridge_list = []
            for bsz, sample in enumerate(mini_batch):
                tc_list = []
                if len(sample.decoder_input_ids_list) > 1:
                    for idx, dec_ids in enumerate(sample.decoder_input_ids_list):
                        if idx == len(sample.decoder_input_ids_list) - 1:
                            continue
                        temp_ids = list_to_tensor([dec_ids], device=device)
                        temp_mask = get_attention_mask(temp_ids, device=device)
                        rep = model.get_time_control_embed(temp_ids, temp_mask)
                        tc_list.append(rep)
                gold_bridge_list.append(tc_list)
            collated_batch = {
                "input": [input_ids, input_mask],
                "decoder_input_all": [decoder_input_all_ids[:, :-1].contiguous(),
                                      decoder_input_all_mask[:, :-1].contiguous()],
                "transition_number": transition_number,
                "gold_bridge_embed": [gold_temp.contiguous(), tc_mask],
                "simulate_bridge_embed": [simulate_temp.contiguous(), tc_mask],
                "user_utterance": [user_utt_ids, user_utt_mask],
                "delta_follow": [delta_follow_ids, delta_follow_mask],
                "start_subgoal": [start_subgoal_ids, start_subgoal_mask],
                "target_subgoal": [target_subgoal_ids, target_subgoal_mask],
                "gold_bridge_list": gold_bridge_list,
            }

        return collated_batch


class PPDPPTorchDataset(BaseTorchDataset):

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        function that preprocess data instances for the PPDPP model
        @param instances: list of data instances.
        @param convert_example_to_feature: the feature converting function
        @return: list of processed data instances.
        """
        processed_instances = []
        for instance in instances:
            # data processing for training the policy model
            feature = convert_example_to_feature(tokenizer=self.tokenizer, instance=instance,
                                                 max_sequence_length=self.max_sequence_length,
                                                 is_test=self.is_test,
                                                 goal2id=self.goal2id
                                                 )
            processed_instances.append(feature)
        return processed_instances

    def collate_fn(self, batch):
        """
        collte function for the PPDPP model
        @param batch: a batch of data instances
        @return: collated batch.
        """
        source_ids, target_ids = zip(*batch)

        input_ids = [torch.tensor(source_id).long() for source_id in source_ids]
        input_ids = torch_pad_sequence(input_ids, batch_first=True, padding_value=0)

        attention_mask = input_ids.ne(0)
        labels = torch.tensor(target_ids).long()

        return {'input_ids': input_ids.to(self.device),
                'attention_mask': attention_mask.to(self.device),
                'labels': labels.to(self.device),
                }
