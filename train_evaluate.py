import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup, T5ForConditionalGeneration, T5Config

from utils import intent_metrics, slot_metrics, convert_t5_output_to_slot_preds, compute_micro_f1, SLOT_MAPPING, \
    ATIS_SLOT_MAPPING


def train_binary(
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        training_params,
        labels_vocab,
        pos_weight: list = None
):
    n_epochs = training_params.epochs
    learning_rate = training_params.learning_rate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    num_training_steps = n_epochs * len(train_dataloader)

    # TODO pos_weight
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([30] * len(labels_vocab)).to(device))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    # criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    progress_bar = trange(num_training_steps, desc="Total Batches")

    best_val_loss = float("inf")
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            batch = tuple(feature.to(device) for feature in batch)
            input_ids, token_type_ids, attention_mask, labels = batch

            logits, probabilities = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            loss = criterion(logits, labels)
            loss.backward()  # accelerator.backward()
            running_loss += loss.item()

            optimizer.step()
            scheduler.step()

            if (i % int(len(train_dataloader) / 4) == 0 or i == len(train_dataloader) - 1) and i != 0:
                print(f" Epoch: {epoch + 1} / {n_epochs}, Training Batch: {i + 1} / {len(train_dataloader)},"
                      f" Loss: {running_loss / i: .3f}")

            progress_bar.update(1)

        val_loss = evaluate_binary(model, val_dataloader, labels_vocab, phase='dev')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # TODO save best model

    if test_dataloader is not None:
        return evaluate_binary(model, test_dataloader, labels_vocab, phase='test')


def evaluate_binary(model, dataloader: DataLoader, labels_vocab, phase='dev'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    preds = []
    labels = []

    for batch in dataloader:
        with torch.no_grad():
            batch = tuple(feature.to(device) for feature in batch)
            input_ids, token_type_ids, attention_mask, batch_labels = batch

            logits, probabilities = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss = criterion(logits, batch_labels)
            running_loss += loss.item()

        # batch_preds = torch.round(probabilities).int()
        # preds.extend(batch_preds.detach().cpu().numpy())
        threshold = 0.5
        batch_preds = torch.where(probabilities > threshold, 1, 0)
        preds.extend(batch_preds.detach().cpu().tolist())

        labels.extend(batch_labels.int().detach().cpu().tolist())

    # preds = np.array(preds).tolist()

    val_loss = running_loss / len(dataloader)
    phase_str = "Validation" if phase == 'dev' else "Test"
    print(f" {phase_str} Loss: {val_loss :.3f}")
    print(f" {phase_str} Accuracy: {accuracy_score(labels, preds) :.3f}")
    # print(f" {phase_str} Weighted F1-score: {f1_score(labels, preds, average='weighted') :.3f}")
    # print(f" {phase_str} Weighted Precision score: {precision_score(labels, preds, average='weighted') :.3f}")
    # print(f" {phase_str} Weighted Recall score: {recall_score(labels, preds, average='weighted') :.3f}")

    print(f" {phase_str} F1-scores: {f1_score(labels, preds, average=None)}")
    print(f" {phase_str} Precision scores: {precision_score(labels, preds, average=None)}")
    print(f" {phase_str} Recall scores: {recall_score(labels, preds, average=None)}")
    return val_loss


def train(
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        training_params,
        intent_labels_vocab,
        slot_labels_vocab,
):
    n_epochs = training_params.epochs
    learning_rate = training_params.learning_rate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    num_training_steps = n_epochs * len(train_dataloader)

    slot_criterion = nn.CrossEntropyLoss()
    intent_criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    progress_bar = trange(num_training_steps, desc="Total Batches")

    best_val_loss = np.float("inf")
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0

        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            batch = tuple(feature.to(device) for feature in batch)
            input_ids, token_type_ids, attention_mask, slot_labels, intent_labels = batch

            intent_out, slot_out = model(
                input_ids=input_ids,
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
                print(f" Epoch: {epoch + 1} / {n_epochs}, Training Batch: {i + 1} / {len(train_dataloader)},"
                      f" Loss: {running_loss / i: .3f}")

            progress_bar.update(1)

        val_loss = evaluate(model, val_dataloader, intent_labels_vocab, slot_labels_vocab, phase='dev')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # TODO save best model

    if test_dataloader is not None:
        return evaluate(model, test_dataloader, intent_labels_vocab, slot_labels_vocab, phase='test')


def evaluate(model, dataloader: DataLoader, intent_labels_vocab, slot_labels_vocab, phase='dev'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    running_loss = 0.0
    intent_criterion, slot_criterion = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()

    slot_preds = []
    slot_labels_true = []
    intent_preds = []
    intent_labels_true = []

    for batch in dataloader:
        with torch.no_grad():
            batch = tuple(feature.to(device) for feature in batch)
            input_ids, token_type_ids, attention_mask, batch_slot_labels, batch_intent_labels = batch

            intent_out, slot_out = model(
                input_ids=input_ids,
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
        slot_preds.append(batch_slot_preds)

        batch_slot_labels = batch_slot_labels.detach().cpu()
        slot_labels_true.append(batch_slot_labels)

        # Intent prediction
        batch_intent_preds = torch.argmax(intent_out, dim=1).detach().cpu().tolist()
        intent_preds.append(batch_intent_preds)

        batch_intent_labels = batch_intent_labels.detach().cpu().tolist()
        intent_labels_true.append(batch_intent_labels)

    val_loss = running_loss / len(dataloader)
    phase_str = "Validation" if phase == 'dev' else "Test"
    print(f" {phase_str} Loss: {val_loss :.3f}")

    # Convert slot_ids (both actual and predictions) to the actual BIO tags in order to feed them to the `seqeval`
    # metrics. Also, ignore tags that belong to the padding token (pad_token_id = -100)
    slot_labels_preds = []
    slot_labels_gold = []
    for batch_true, batch_pred in zip(slot_labels_true, slot_preds):
        for utter_true, utter_pred in zip(batch_true, batch_pred):
            slot_labels_preds.append(
                [slot_labels_vocab[slot_pred_idx]
                 for slot_true_idx, slot_pred_idx in zip(utter_true, utter_pred)
                 if slot_true_idx != slot_criterion.ignore_index]
            )
            slot_labels_gold.append(
                [slot_labels_vocab[slot_true_idx] for slot_true_idx in utter_true
                 if slot_true_idx != slot_criterion.ignore_index]
            )

    slot_f1 = slot_metrics(slot_labels_gold, slot_labels_preds)

    intent_preds = [intent_pred for batch in intent_preds for intent_pred in batch]
    intent_labels_true = [intent_true for batch in intent_labels_true for intent_true in batch]
    intent_accuracy, intent_f1 = intent_metrics(intent_labels_true, intent_preds)

    print(f" {phase_str} Intent Accuracy: {intent_accuracy: .3f}, {phase_str} Intent F1: {intent_f1: .3f},"
          f" {phase_str} Slot F1: {slot_f1: .3f}\n")

    if phase == 'test':
        intent_labels_preds = [intent_labels_vocab[intent_pred] for intent_pred in intent_preds]
        return intent_labels_preds, slot_labels_preds
    else:
        return val_loss


def train_seq2seq_model(
        model,
        tokenizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        model_weights_path='./t5-small_parameters.pt'
):
    n_epochs = 30
    learning_rate = 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    num_training_steps = n_epochs * len(train_dataloader)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    progress_bar = trange(num_training_steps, desc="Total Batches")

    best_val_f1 = 0.0
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            running_loss += loss.item()

            optimizer.step()
            scheduler.step()

            if (i % int(len(train_dataloader) / 4) == 0 or i == len(train_dataloader) - 1) and i != 0:
                print(f" Epoch: {epoch + 1} / {n_epochs}, Training Batch: {i + 1} / {len(train_dataloader)},"
                      f" Loss: {running_loss / i: .3f}, Learning Rate: {scheduler.get_last_lr()[0]: .5f}")

            progress_bar.update(1)

        if val_dataloader is not None:
            val_f1 = evaluate_seq2seq_model(model, tokenizer, val_dataloader, phase='dev')
            if model_weights_path is not None and val_f1 > best_val_f1:
                torch.save(model.state_dict(), model_weights_path)
                best_val_f1 = val_f1

    if test_dataloader is not None:
        checkpoint = 't5-small'
        config = T5Config.from_pretrained(checkpoint)
        model = T5ForConditionalGeneration(config=config)
        model.load_state_dict(torch.load(model_weights_path))
        _ = evaluate_seq2seq_model(model, tokenizer, test_dataloader, phase='test')


def evaluate_seq2seq_model(model, tokenizer, dataloader, phase='dev'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    running_loss = 0.0

    preds_lst = []
    labels_lst = []

    for batch in dataloader:
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

            gen_kwargs = {"max_length": 300, "early_stopping": False}
            generated_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

            # Replace -100 in the labels as we can't decode them.
            labels[labels == nn.CrossEntropyLoss().ignore_index] = model.config.pad_token_id

            preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

            preds_lst.extend(preds)
            labels_lst.extend(labels)

    cleaned_preds = [convert_t5_output_to_slot_preds(pred) for pred in preds_lst]
    cleaned_labels = [convert_t5_output_to_slot_preds(label) for label in labels_lst]

    assert len(cleaned_preds) == len(cleaned_labels)
    # for pr, lbl in zip(cleaned_preds, cleaned_labels):
    #     if len(pr) != len(lbl):
    #         print(f'Not equal lengths. Difference of {(len(pr)-len(lbl))} slots')

    # F1 calculation
    scores = {}
    mapping = SLOT_MAPPING if dataloader.dataset.dataset == 'snips' else ATIS_SLOT_MAPPING
    for slot in mapping:
        scores[slot] = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0
        }

    # acc is defined as how many of the gold labels does the model find regardless of order and duplicates
    acc = 0.0

    for i, (test_example_preds, test_example_labels) in enumerate(zip(cleaned_preds, cleaned_labels)):
        dataset = dataloader.dataset
        #################
        # for 2-intent few-shot setting (should be removed otherwise)
        # subset = dataloader.dataset
        # dataset = subset.dataset
        # i = subset.indices[i]

        # assert intent in ['AddToPlaylist', 'BookRestaurant']
        #################

        intent = dataset.intents[i]
        intent_slot_mapping = dataset.intent_slots_mapping[intent]

        curr_preds = [val.strip() for val in test_example_preds if val != ' none']
        curr_labels = [val.strip() for val in test_example_labels if val != ' none']

        if len(curr_labels) != 0:
            acc += sum(1 for lbl in curr_labels if lbl in curr_preds) / len(curr_labels)

        # for slot_name, slot_pred, gold_slot in zip(intent_slot_mapping, test_example_preds, test_example_labels):
        assert len(intent_slot_mapping) == len(test_example_labels)
        for j, (slot_name, gold_slot) in enumerate(zip(intent_slot_mapping, test_example_labels)):
            try:
                slot_pred = test_example_preds[j]
            except IndexError:
                slot_pred = 'none'
            slot_pred, gold_slot = slot_pred.strip(), gold_slot.strip()
            if slot_pred != 'none':
                if gold_slot == 'none':
                    # print(f"Test example {i+1}\t{slot_name=}\t{slot_pred=}")
                    scores[slot_name]["false_positives"] += 1
                else:
                    if gold_slot == slot_pred:
                        # print(f"TPTest example {i + 1}\t{slot_name=}\t{slot_pred=}\t{gold_slot=}")
                        scores[slot_name]["true_positives"] += 1
                    else:
                        # print(f"Test example {i + 1}\t{slot_name=}\t{slot_pred=}\t{gold_slot=}")
                        # Maybe skip the following?
                        scores[slot_name]["false_negatives"] += 1
            else:
                if gold_slot != 'none':
                    # print(f"Test example {i + 1}\t{slot_name=}\t{slot_pred=}\t{gold_slot=}")
                    scores[slot_name]["false_negatives"] += 1

    phase_str = "Validation" if phase == 'dev' else "Test"
    print(f"\n {phase_str} Set Results:")

    prec, rec, f1 = compute_micro_f1(scores=scores)
    print(f"Micro F1 score = {f1: .3f}\nMicro Precision score = {prec: .3f}\nMicro Recall score = {rec: .3f}\n"
          f"Accuracy = {acc / len(cleaned_preds): .3f}")

    return f1
