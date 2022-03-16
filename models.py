import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, T5ForConditionalGeneration


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
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        slots_output = outputs[0]
        intent_output = outputs[1]

        intent_logits = self.intent_head(self.dropout(intent_output))
        slot_logits = self.slot_head(self.dropout(slots_output))

        return intent_logits, slot_logits


class T5PromptTuningMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        initialize_from_vocab: bool = False,
        random_range: float = 0.5,
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Make sure to freeze initial T5 model
        T5PromptTuningMixin.freeze_model_params(model)

        print("Initializing soft prompt...")
        model.initialize_soft_prompt(
            initialize_from_vocab=initialize_from_vocab, random_range=random_range
        )

        return model

    @staticmethod
    def freeze_model_params(model):
        for param in model.parameters():
            param.requires_grad = False

    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initialize_soft_prompt(
        self,
        n_tokens: int = 20,
        initialize_from_vocab: bool = False,
        random_range: float = 0.5,
    ) -> None:
        self.n_tokens = n_tokens
        if initialize_from_vocab:
            init_prompt_value = self.shared.weight[:n_tokens].clone().detach()
        else:
            init_prompt_value = torch.FloatTensor(n_tokens, self.model_dim).uniform_(
                -random_range, random_range
            )

        self.soft_prompt = nn.Embedding(n_tokens, self.model_dim)
        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        inputs_embeds = self.shared(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)
        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]

        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1).to(self.device), attention_mask],
            dim=1,
        )

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        kwargs["input_ids"] = kwargs["input_ids"].to(self.device)
        if kwargs["input_ids"] is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(
                kwargs["input_ids"]
            ).to(self.device)

        if kwargs["attention_mask"] is not None:
            attention_mask = self._extend_attention_mask(kwargs["attention_mask"]).to(
                self.device
            )

        return super().generate(inputs_embeds, attention_mask)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        return_dict=None,
    ):
        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(
                self.device
            )
        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(self.device)

        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )


class T5PromptTuningLM(T5PromptTuningMixin, T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
