import argparse

import torch.cuda
from torch.utils.data.dataset import Subset
from omegaconf import OmegaConf
from transformers import TrainingArguments, Trainer, BartForConditionalGeneration, BartTokenizer

from joint_dataset import get_dataloader, IntentDataset, SlotDataset
from models import JointBert
from train import train
from utils import set_seed, intent_metrics_bart, INTENT_MAPPING, convert_output_to_slot_preds, SLOT_MAPPING,\
    compute_micro_f1
from reader import Reader


def main(run_args, model_config):
    torch.cuda.empty_cache()
    set_seed(run_args.seed)

    reader = Reader(run_args.dataset)
    reader.construct_intent_and_slot_vocabs(write_to_disk=True)

    intent_labels, slot_labels = reader.get_intent_labels(), reader.get_slot_labels()

    if run_args.approach.lower() == 'fine-tune' and run_args.model_type.lower() == 'bert':
        model = JointBert(model_config, len(intent_labels), len(slot_labels))

        if run_args.do_train and run_args.do_eval:
            train_dataloader = get_dataloader(run_args.dataset,
                                              mode='train',
                                              batch_size=model_config.batch_size,
                                              model_name=model_config.model)
            val_dataloader = get_dataloader(run_args.dataset,
                                            mode='dev',
                                            batch_size=model_config.batch_size,
                                            model_name=model_config.model)
            test_dataloader = get_dataloader(run_args.dataset,
                                             mode='test',
                                             batch_size=model_config.batch_size,
                                             model_name=model_config.model)
            if run_args.save_preds:
                intent_preds, slot_preds = train(model,
                                                 train_dataloader,
                                                 val_dataloader,
                                                 test_dataloader,
                                                 model_config,
                                                 intent_labels,
                                                 slot_labels
                                                 )
                reader.save_test_preds_to_csv(intent_preds, slot_preds)

            else:
                train(model, train_dataloader, val_dataloader, test_dataloader, model_config, intent_labels,
                      slot_labels)

    elif run_args.approach.lower() == 'prompt' and run_args.model_type.lower() == 'bart':
        checkpoint = 'facebook/bart-base'
        tokenizer = BartTokenizer.from_pretrained(checkpoint)
        checkpoint = './test-bart/checkpoint-2000'
        model = BartForConditionalGeneration.from_pretrained(checkpoint, forced_bos_token_id=tokenizer.bos_token_id)

        # train_dataset = IntentDataset(dataset=run_args.dataset, mode='train', tokenizer=tokenizer)
        # val_dataset = IntentDataset(dataset=run_args.dataset, mode='dev', tokenizer=tokenizer)
        # test_dataset = IntentDataset(dataset=run_args.dataset, mode='test', tokenizer=tokenizer)

        train_dataset = SlotDataset(dataset=run_args.dataset, mode='train', tokenizer=tokenizer)
        val_dataset = SlotDataset(dataset=run_args.dataset, mode='dev', tokenizer=tokenizer)
        test_dataset = SlotDataset(dataset=run_args.dataset, mode='test', tokenizer=tokenizer)

        #############
        # Try out a subset of the training set (few-shot)
        # indices = torch.randperm(len(train_dataset))[:1000]
        # train_dataset = Subset(train_dataset, indices=indices)

        # Use Huggingface's Trainer class for training and evaluation
        training_args = TrainingArguments(
            output_dir="test-bart-without-mask",
            # evaluation_strategy="epoch",
            dataloader_num_workers=4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=1,
            num_train_epochs=2,
            load_best_model_at_end=False,
            # eval_accumulation_steps=4
        )

        trainer = Trainer(
            model,
            training_args,
            train_dataset=train_dataset,
            # eval_dataset=val_dataset,
            # compute_metrics=intent_metrics_bart
        )

        # trainer.train()
        # trainer.evaluate()

        # predict_results = trainer.predict(test_dataset)
        # Select a subset of the test dataset for evaluation because the whole set does not fit into GPU memory
        begin_prediction_range = 500
        num_predictions = 100

        to_predict = [test_dataset[i] for i in range(begin_prediction_range, begin_prediction_range + num_predictions)]
        predict_results = trainer.predict(to_predict)

        predictions = tokenizer.batch_decode(
            predict_results.predictions[0].argmax(axis=-1),
            skip_special_tokens=False
        )

        # initialize scores for calculating micro-f1
        scores = {}
        for slot in SLOT_MAPPING:
            scores[slot] = {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0
            }

        acc_per_utter = []
        for i in range(begin_prediction_range, begin_prediction_range + num_predictions):
            utter = test_dataset.sentences[i]
            tags = test_dataset.slots[i]

            gold_slots = SlotDataset.get_clean_slots_dict(utter, tags)
            pred_slots = convert_output_to_slot_preds(predictions[i-begin_prediction_range])

            # For error inspecting
            # intent = test_dataset.intents[i]
            # intent_slot_mapping = test_dataset.intent_slots_mapping[intent]
            # if list(intent_slot_mapping) != list(pred_slots.keys()):
            #     print(intent_slot_mapping, '\n', pred_slots.keys())

            for slot_name, slot_pred in pred_slots.items():
                gold_slot = gold_slots.get(slot_name, '')
                if slot_pred != 'none':
                    if gold_slot == '':
                        scores[slot_name]["false_positives"] += 1
                        print(f"Test example {i+1}\t{slot_name=}\t{slot_pred=}")
                    else:
                        if gold_slot == slot_pred:
                            scores[slot_name]["true_positives"] += 1
                        else:
                            # Maybe skip the following?
                            scores[slot_name]["false_negatives"] += 1
                else:
                    if gold_slot != '':
                        print(f"Test example {i + 1}\t{slot_name=}\t{slot_pred=}\t{gold_slot=}")
                        scores[slot_name]["false_negatives"] += 1

        prec, rec, f1 = compute_micro_f1(scores=scores)
        print(f"Micro F1 score = {f1: .3f}\nMicro Precision score = {prec: .3f}\nMicro Recall score = {rec: .3f}")

        # Evaluation for the IntentDataset

        # gold_intents = [INTENT_MAPPING[intent] for intent in test_dataset.intents]
        # correct_preds = sum(1 for pred, label in zip(predictions, gold_intents) if pred.split()[-1] == label)
        # accuracy = correct_preds / len(test_dataset)
        # accuracy = correct_preds / len(to_predict)
        # print(f"\n{accuracy = }\n")
        #
        # # Error analysis
        # for pred, label, init_label in zip(predictions, gold_intents, test_dataset.intents):
        #     if pred.split()[-1] != label:
        #         print(f"True label: {init_label: >15} \tMapped label: {label: >15}\tPrediction: {pred}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--approach", default="prompt", type=str, help="Select approach between prompt and fine-tune")
    # parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    # parser.add_argument("--data_dir", default="./data", type=str, help="The input data directory")
    parser.add_argument("--dataset", default="snips", type=str, help="The input dataset")
    parser.add_argument("--model_type", default="bart", type=str, help="Select model type")
    parser.add_argument('--seed', type=int, default=42, help="Seed for reproducibility")

    parser.add_argument('--do_train', default=True, type=bool, help="Whether to train the model.")
    parser.add_argument("--do_eval", default=True, type=bool, help="Whether to evaluate the model on the test set.")

    parser.add_argument("--save_preds", default=True, type=bool, help="Whether to save model's predictions to csv.")

    run_args = parser.parse_args()
    model_config = OmegaConf.load("config/model_config.yaml")

    main(run_args, model_config)
