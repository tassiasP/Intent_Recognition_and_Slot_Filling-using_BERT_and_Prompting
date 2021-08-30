import torch.nn as nn
from transformers import BertConfig, BertModel


class JointBert(nn.Module):

    def __init__(self, config, num_intent_labels, num_slot_labels):
        super(JointBert, self).__init__()
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels

        self.bert_conf = BertConfig.from_pretrained(config.model)
        self.bert = BertModel.from_pretrained(config.model)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.intent_head = nn.Linear(self.bert_conf.hidden_size, self.num_intent_labels)
        self.slot_head = nn.Linear(self.bert_conf.hidden_size, self.num_slot_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        slots_output = outputs[0]
        intent_output = outputs[1]

        intent_logits = self.intent_head(self.dropout(intent_output))
        slot_logits = self.slot_head(self.dropout(slots_output))

        return intent_logits, slot_logits



