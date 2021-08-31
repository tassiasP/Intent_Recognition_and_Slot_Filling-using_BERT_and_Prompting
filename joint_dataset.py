import copy
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn

from transformers import BertTokenizerFast, BatchEncoding
from transformers.hf_argparser import HfArgumentParser

from reader import Reader


@dataclass(frozen=True)
class InputExample:
    id: int
    utterance: List[str]
    slot_labels: List[str]
    intent_label: str

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
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


def get_dataloader(dataset: str, mode: str, batch_size: int, model_name: str) -> DataLoader:
    processor = JointProcessor(model_name)
    dataset = JointDataset(dataset=dataset, mode=mode, processor=processor)

    sampler = RandomSampler(data_source=dataset) if mode == 'train' else SequentialSampler(data_source=dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=8)

    return dataloader
