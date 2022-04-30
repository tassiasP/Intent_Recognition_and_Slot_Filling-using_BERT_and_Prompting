import copy
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataset import Subset, T_co
import torch.nn as nn
from torch.nn.functional import one_hot
from transformers import BertTokenizerFast, BatchEncoding, T5Tokenizer

from reader import Reader
from utils import INTENT_MAPPING, SLOT_MAPPING, ATIS_INTENT_MAPPING, ATIS_SLOT_MAPPING


@dataclass(frozen=True)
class InputExample:
    id: int
    utterance: List[str]
    slot_labels: List[str]
    intent_label: str

    def to_dict(self):
        return copy.deepcopy(self.__dict__)


class JointProcessor:
    def __init__(self, model_name: str, max_len=64):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.max_len = max_len

    def convert_example_to_bert_features(self, utterance, intent_id, slot_ids):
        # tokenized_string = self.tokenizer.tokenize(utterance)
        tokenized_utterance: BatchEncoding = self.tokenizer(utterance,
                                                            padding="max_length",
                                                            max_length=self.max_len,
                                                            truncation=True,
                                                            is_split_into_words=True
                                                            )
        # ignore slot label = -100
        ignore_id = nn.CrossEntropyLoss().ignore_index
        slot_labels = []

        word_ids = tokenized_utterance.word_ids()
        previous_word_idx = None

        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                slot_labels.append(ignore_id)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                slot_labels.append(slot_ids[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                slot_labels.append(ignore_id)

            previous_word_idx = word_idx

        input_ids = torch.tensor(tokenized_utterance["input_ids"], dtype=torch.long)
        token_type_ids = torch.tensor(tokenized_utterance["token_type_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized_utterance["attention_mask"], dtype=torch.long)
        slot_labels = torch.tensor(slot_labels, dtype=torch.long)
        intent_label = torch.tensor(intent_id, dtype=torch.long)

        return input_ids, token_type_ids, attention_mask, slot_labels, intent_label


class JointDataset(Dataset):

    def __init__(self, dataset: str, mode: str, processor: JointProcessor):
        super().__init__()
        self.dataset = dataset
        self.mode = mode

        reader = Reader(self.dataset)
        self.sentences, self.slots, self.intents = reader.read_dataset(mode=self.mode)
        # TODO for guid, (sent, slots, intent) in enumerate reader.read_dataset(): ex = InputExample(...)

        self.intent_labels = reader.get_intent_labels()
        self.slot_labels = reader.get_slot_labels()

        self.processor = processor

    def get_intent_id(self, example: InputExample):
        if example.intent_label in self.intent_labels:
            return self.intent_labels.index(example.intent_label)
        else:
            return self.intent_labels.index("UNK")

    def get_slots_ids(self, example: InputExample):
        slot_ids = []
        for slot in example.slot_labels:
            if slot in self.slot_labels:
                slot_ids.append(self.slot_labels.index(slot))
            else:
                slot_ids.append(self.slot_labels.index("UNK"))

        return slot_ids

    def __len__(self):
        return len(self.intents)

    def __getitem__(self, index):
        # TODO define getter in `InputExample` based on index
        example = InputExample(index, self.sentences[index], self.slots[index], self.intents[index])
        intent_id = self.get_intent_id(example)
        slot_ids = self.get_slots_ids(example)

        return self.processor.convert_example_to_bert_features(example.utterance, intent_id, slot_ids)


def get_dataloader(dataset_name: str, mode: str, batch_size: int, model_name: str) -> DataLoader:
    processor = JointProcessor(model_name)
    dataset = JointDataset(dataset=dataset_name, mode=mode, processor=processor)

    # Try out a subset of the training set (few-shot)
    if mode == 'train':
        indices = torch.randperm(len(dataset))[:200]
        dataset = Subset(dataset, indices=indices)

    sampler = RandomSampler(data_source=dataset) if mode == 'train' else SequentialSampler(data_source=dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=8)

    return dataloader


class IntentDataset(Dataset):
    def __init__(self, dataset: str, mode: str, tokenizer, max_len=64, use_prompting: bool = True):
        super().__init__()
        self.dataset = dataset
        self.mode = mode
        self.max_len = max_len
        self.use_prompting = use_prompting

        self.tokenizer = tokenizer

        self.reader = Reader(self.dataset)
        self.sentences, self.slots, self.intents = self.reader.read_dataset(mode=self.mode)

    def convert_utters_to_model_inputs(self, input_utter, output_utter):
        """Utilizes tokenizer to convert tokens to ids that will be fed to the model"""
        model_inputs = self.tokenizer(
            input_utter,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            is_split_into_words=True,
            return_tensors='pt'
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                output_utter,
                padding="max_length",
                max_length=int(self.max_len / 2),
                # max_length=int(self.max_len),
                truncation=True,
                is_split_into_words=True,
                return_tensors='pt'
            )

        labels = labels["input_ids"]
        # Replace pad token id with -100 in order to ignore padding tokens during loss calculation
        labels[labels == self.tokenizer.pad_token_id] = nn.CrossEntropyLoss().ignore_index

        model_inputs["labels"] = labels
        model_inputs = {k: feature.flatten() for k, feature in model_inputs.items()}

        return model_inputs

    def __len__(self):
        return len(self.intents)

    def __getitem__(self, index):
        utter = self.sentences[index]
        intent = self.intents[index]

        input_utter = copy.deepcopy(utter)
        if self.use_prompting:
            prompt = '. This sentence refers to <extra_id_0>'
            # prompt = '. This intent is related to <extra_id_0>'
            input_utter.extend(prompt.split())
        else:
            extra_tokens = '. <extra_id_0>'
            input_utter.extend(extra_tokens.split())

        output_utter = '<extra_id_0> ' + INTENT_MAPPING[intent] + ' <extra_id_1>' \
            if self.dataset == 'snips' \
            else '<extra_id_0> ' + ATIS_INTENT_MAPPING[intent] + ' <extra_id_1>'
        output_utter = output_utter.split()

        return self.convert_utters_to_model_inputs(input_utter=input_utter, output_utter=output_utter)


class SlotDataset(IntentDataset):
    def __init__(self, dataset: str, mode: str, tokenizer: T5Tokenizer, max_len: int = 500, use_prompting: bool = True):
        super().__init__(dataset, mode, tokenizer, max_len, use_prompting)
        self.intent_slots_mapping = self._get_intent_slots_mapping()

    def _get_intent_slots_mapping(self):
        """ Returns a dictionary which maps each intent to the relevant slots based on the training set"""
        intent_slots_mapping = defaultdict(set)
        _, slots, intents = self.reader.read_dataset(mode='train')
        assert len(slots) == len(intents)

        for tags, intent in zip(slots, intents):
            for tag in tags:
                if tag != 'O':
                    intent_slots_mapping[intent].add(tag[2:])

        # Sort slot names to provide the template in specific order
        intent_slots_mapping_sorted = {intent_name: sorted(slot_names)
                                       for intent_name, slot_names in intent_slots_mapping.items()}

        return intent_slots_mapping_sorted

    @staticmethod
    def get_clean_slots_dict(utter: List[str], tags: List[str]):
        # example:
        #   utter = ["listen", "to", "westbam", "alumb", "allergic", "on", "google", "music"]
        #   tags = ["O", "O", "B-artist", "O", "B-album", "O", "B-service", "I-service"]
        #   returns: dict({"artist": "westbam", "album": "allergic", "service": "google music"})

        slots = {}
        prev_bio_tag = 'O'

        for word, tag in zip(utter, tags):
            if tag.startswith('B-'):
                if prev_bio_tag == 'B' or prev_bio_tag == 'I':
                    slots[slot_key] = span
                slot_key = tag[2:]
                span = word
                prev_bio_tag = 'B'
            elif tag.startswith('I-'):
                span += ' ' + word
                prev_bio_tag = 'I'
            else:  # it is an 'O'
                if prev_bio_tag != 'O':
                    slots[slot_key] = span
                prev_bio_tag = 'O'
        else:
            if prev_bio_tag != 'O':
                slots[slot_key] = span

        return slots

    def get_template(self, relevant_slots: set[str], slots_dict_without_bio: dict):
        example_slots = {slot for slot in slots_dict_without_bio}
        try:
            assert example_slots.issubset(relevant_slots)
        except AssertionError:
            pass
            # print([slot for slot in example_slots if slot not in relevant_slots])

        eos_token = self.tokenizer.eos_token
        sep_token = self.tokenizer.sep_token
        separator = '.'  # sep_token

        if self.use_prompting:
            input_template = '. '
            output_template = ''

            for slot_num, slot in enumerate(relevant_slots):
                slot_name: str = SLOT_MAPPING[slot] if self.dataset == 'snips' else ATIS_SLOT_MAPPING[slot]
                if slot_num != (len(relevant_slots) - 1):
                    input_template += f"The {slot_name} is <extra_id_{slot_num}>{separator}"
                    if slot in example_slots:
                        output_template += f"<extra_id_{slot_num}> {slots_dict_without_bio[slot]} "
                    else:
                        output_template += f"<extra_id_{slot_num}> none "
                else:  # if at last iteration also add <extra_id> (sentinel token) at the end of the output
                    input_template += f"The {slot_name} is <extra_id_{slot_num}>"
                    if slot in example_slots:
                        output_template += f"<extra_id_{slot_num}> {slots_dict_without_bio[slot]} <extra_id_{slot_num + 1}>"
                    else:
                        output_template += f"<extra_id_{slot_num}> none <extra_id_{slot_num + 1}>"
        else:
            input_template = ''
            output_template = ''

            for slot_num, slot in enumerate(relevant_slots):
                if slot_num != (len(relevant_slots) - 1):
                    input_template += f"<extra_id_{slot_num}> "
                    if slot in example_slots:
                        output_template += f"<extra_id_{slot_num}> {slots_dict_without_bio[slot]} "
                    else:
                        output_template += f"<extra_id_{slot_num}> none "
                else:  # if at last iteration also add <extra_id> (sentinel token) at the end
                    input_template += f"<extra_id_{slot_num}> "
                    if slot in example_slots:
                        output_template += f"<extra_id_{slot_num}> {slots_dict_without_bio[slot]} <extra_id_{slot_num + 1}>"
                    else:
                        output_template += f"<extra_id_{slot_num}> none <extra_id_{slot_num + 1}>"

        return input_template, output_template

    def __getitem__(self, index):
        utter = self.sentences[index]
        intent = self.intents[index]
        slots = self.slots[index]

        relevant_slots = self.intent_slots_mapping[intent]
        slots_dict_without_bio = self.get_clean_slots_dict(utter, slots)

        input_template, output_template = self.get_template(relevant_slots, slots_dict_without_bio)

        input_utter = copy.deepcopy(utter)
        input_utter.extend(input_template.split())

        output_utter = output_template.split()

        return self.convert_utters_to_model_inputs(input_utter=input_utter, output_utter=output_utter)

    # Uncomment to try out few-shot setting
    # def __len__(self):
    #     if self.mode == 'train':
    #         return 500
    #     else:
    #         return len(self.intents)


class BinaryDataset(JointDataset):

    def __init__(self, dataset: str, mode: str):
        self.processor = JointProcessor(model_name="bert-base-uncased")
        super().__init__(dataset, mode, self.processor)

        self.slot_labels = self.remove_bio(self.slot_labels)

    @staticmethod
    def remove_bio(slot_labels):
        cleaned_labels = {label[2:] for label in slot_labels if label != 'O' and label not in ["UNK", "PAD"]}
        cleaned_labels.add("UNK")
        return sorted(list(cleaned_labels))

    def get_slots_ids(self, example: InputExample):
        slot_ids = set()
        for slot in example.slot_labels:
            if slot != 'O':
                slot = slot[2:]
                if slot in self.slot_labels:
                    slot_ids.add(self.slot_labels.index(slot))
                else:
                    slot_ids.add(self.slot_labels.index("UNK"))

        return list(slot_ids)

    def __getitem__(self, index):
        example = InputExample(index, self.sentences[index], self.slots[index], self.intents[index])
        model_inputs: BatchEncoding = self.processor.tokenizer(
            example.utterance,
            padding="max_length",
            max_length=self.processor.max_len,
            truncation=True,
            is_split_into_words=True
        )
        # one hot encode the labels
        slot_ids = self.get_slots_ids(example)
        slot_ids = one_hot(torch.tensor(slot_ids, dtype=torch.int64), len(self.slot_labels))
        labels = slot_ids.sum(dim=0).float()
        # model_inputs["labels"] = torch.unsqueeze(slot_ids, dim=0)

        input_ids = torch.tensor(model_inputs["input_ids"], dtype=torch.long)
        token_type_ids = torch.tensor(model_inputs["token_type_ids"], dtype=torch.long)
        attention_mask = torch.tensor(model_inputs["attention_mask"], dtype=torch.long)

        # labels = slot_ids.clone().detach()
        # labels = torch.tensor(slot_ids, dtype=torch.float)
        # slot_labels = torch.tensor(slot_labels, dtype=torch.long)
        # intent_label = torch.tensor(intent_id, dtype=torch.long)

        return input_ids, token_type_ids, attention_mask, labels


def get_binary_dataloader(dataset_name: str, mode: str, batch_size: int) -> DataLoader:
    dataset = BinaryDataset(dataset=dataset_name, mode=mode)

    # Try out a subset of the training set (few-shot)
    if mode == 'train':
        indices = torch.randperm(len(dataset))[:500]
        dataset = Subset(dataset, indices=indices)

    sampler = RandomSampler(data_source=dataset) if mode == 'train' else SequentialSampler(data_source=dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=8)

    return dataloader
