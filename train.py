from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import AdamW, get_linear_schedule_with_warmup

from models import JointBert


def train(model: JointBert, train_dataloader: DataLoader, training_params, intent_labels_vocab, slot_labels_vocab):
    n_epochs = training_params.epochs
    learning_rate = training_params.learning_rate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    num_training_steps = n_epochs * len(train_dataloader)

    slot_criterion = nn.CrossEntropyLoss()
    intent_criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    progress_bar = trange(num_training_steps)
    epoch_iter = trange(n_epochs)
    for epoch in epoch_iter:
        model.train()
        running_loss = 0.0

        train_iter = tqdm(train_dataloader)
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()

            batch = tuple(feature.to(device) for feature in batch)
            input_ids, token_type_ids, attention_mask, slot_labels, intent_labels = batch

            intent_out, slot_out = model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids
                                         )

            active_loss = attention_mask.view(-1) == 1
            active_logits = slot_out.view(-1, len(slot_labels_vocab))[active_loss]
            active_labels = slot_labels.view(-1)[active_loss]
            slot_loss = slot_criterion(active_logits, active_labels)

            intent_loss = intent_criterion(
                intent_out.view(-1, len(intent_labels_vocab)),
                intent_labels.view(-1),
            )

            loss = slot_loss + intent_loss
            loss.backward()  # accelerator.backward()
            running_loss += loss.item()

            optimizer.step()
            scheduler.step()

            if (i % 50 == 0 or i == len(train_dataloader)-1) and i != 0:
                print(f"Epoch: {epoch}, Training Steps: {i+1} / {len(train_iter)}, Loss: {running_loss / i}")

            progress_bar.update(32)
