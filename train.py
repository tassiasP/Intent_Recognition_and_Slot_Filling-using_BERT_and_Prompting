import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import AdamW, get_linear_schedule_with_warmup

from utils import intent_metrics, slot_metrics


def train(model, train_dataloader: DataLoader, val_dataloader, training_params, intent_labels_vocab, slot_labels_vocab):
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
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_dataloader):
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
                intent_labels.view(-1)
            )

            loss = slot_loss + intent_loss
            loss.backward()  # accelerator.backward()
            running_loss += loss.item()

            optimizer.step()
            scheduler.step()

            if (i % int(len(train_dataloader) / 4) == 0 or i == len(train_dataloader) - 1) and i != 0:
                print(f" Epoch: {epoch + 1}, Training Batch: {i + 1} / {len(train_dataloader)},"
                      f" Loss: {running_loss / i: .3f}")

            progress_bar.update(1)

        evaluate(model, val_dataloader, intent_labels_vocab, slot_labels_vocab)


def evaluate(model, dataloader: DataLoader, intent_labels_vocab, slot_labels_vocab):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    running_loss = 0.0
    intent_criterion, slot_criterion = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()

    slot_preds = list()  # np.array([])
    slot_labels_true = list()  # np.array([])

    for batch in dataloader:
        with torch.no_grad():
            batch = tuple(feature.to(device) for feature in batch)
            input_ids, token_type_ids, attention_mask, batch_slot_labels, batch_intent_labels = batch

            intent_out, slot_out = model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids
                                         )

            # Loss calculation
            active_loss = attention_mask.view(-1) == 1
            active_logits = slot_out.view(-1, len(slot_labels_vocab))[active_loss]
            active_labels = batch_slot_labels.view(-1)[active_loss]

            slot_loss = slot_criterion(active_logits, active_labels)
            intent_loss = intent_criterion(
                intent_out.view(-1, len(intent_labels_vocab)),
                batch_intent_labels.view(-1)
            )

            loss = slot_loss + intent_loss
            running_loss += loss.item()

        # Slot prediction
        batch_slot_preds = torch.argmax(slot_out, dim=2).detach().cpu()
        # slot_preds = np.concatenate(slot_preds, batch_slot_preds)
        slot_preds.append(batch_slot_preds)

        batch_slot_labels = batch_slot_labels.detach().cpu()
        # slot_labels_true = np.concatenate(slot_labels_true, batch_slot_labels)
        slot_labels_true.append(batch_slot_labels)


    print(f" Validation Loss: {running_loss / len(dataloader) :.3f}")

    # slot_labels_preds = [[slot_labels_vocab[slot_pred_idx] for slot_pred_idx in utter]
    #                      for batch in slot_preds
    #                      for utter in batch
    #                      if slot_criterion.ignore_index]
    # slot_labels_true = [[slot_labels_vocab[slot_true_idx] for slot_true_idx in utter] for batch in slot_labels_true for utter in batch]

    slot_labels_preds = []
    slot_labels_gold = []
    for batch_true, batch_pred in zip(slot_labels_true, slot_labels_preds):
        for utter_true, utter_pred in zip(batch_true, batch_pred):
            for slot_true_idx, slot_pred_idx in zip(utter_true, utter_pred):
                if slot_true_idx != slot_criterion.ignore_index:
                    slot_labels_preds.append(slot_labels_vocab[slot_pred_idx])
                    slot_labels_gold.append(slot_labels_vocab[slot_true_idx])

    slot_f1 = slot_metrics(slot_labels_gold, slot_labels_preds)

    intent_preds = torch.argmax(intent_out, dim=1).detach().cpu().numpy()
    intent_labels_true = batch_intent_labels.detach().cpu().numpy()
    intent_accuracy, intent_f1 = intent_metrics(intent_labels_true, intent_preds)
    print(f" Validation Intent Accuracy: {intent_accuracy}, Validation Intent F1: {intent_f1: .2f},"
          f" Validation Slot F1: {slot_f1}\n")
