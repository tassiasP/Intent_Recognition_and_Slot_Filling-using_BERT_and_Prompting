import argparse

import torch.cuda
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from omegaconf import OmegaConf
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer, T5Config, T5Tokenizer, T5ForConditionalGeneration
from transformers import logging

from joint_dataset import get_dataloader, IntentDataset, SlotDataset
from models import JointBert
from train_evaluate import train, train_seq2seq_model
from utils import set_seed, INTENT_MAPPING, convert_bart_output_to_slot_preds, SLOT_MAPPING, compute_micro_f1
from reader import Reader


def main(run_args, model_config):
    torch.cuda.empty_cache()
    set_seed(run_args.seed)
    logging.set_verbosity(logging.CRITICAL)

    reader = Reader(run_args.dataset)
    reader.construct_intent_and_slot_vocabs(write_to_disk=True)

    intent_labels, slot_labels = reader.get_intent_labels(), reader.get_slot_labels()

    if run_args.approach.lower() == 'fine-tuning' and run_args.model_type.lower() == 'bert':
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

    elif run_args.approach.lower() == 'prompting':
        if run_args.model_type.lower() == 'bart':
            checkpoint = 'facebook/bart-base'

            config = BartConfig.from_pretrained(checkpoint)
            config.force_bos_token_to_be_generated = True
            # config.forced_bos_token_id = tokenizer.bos_token_id

            tokenizer = BartTokenizer.from_pretrained(checkpoint)

            checkpoint = './test-bart-raw-slots/checkpoint-3000'
            model = BartForConditionalGeneration.from_pretrained(checkpoint, config=config)
        elif run_args.model_type.lower() == 't5':
            checkpoint = 't5-small'

            # config = T5Config.from_pretrained(checkpoint)
            tokenizer = T5Tokenizer.from_pretrained(checkpoint)
            checkpoint = './test-t5/checkpoint-3000'
            model = T5ForConditionalGeneration.from_pretrained(checkpoint)

        train_dataset = SlotDataset(dataset=run_args.dataset, mode='train', tokenizer=tokenizer)
        val_dataset = SlotDataset(dataset=run_args.dataset, mode='dev', tokenizer=tokenizer)
        test_dataset = SlotDataset(dataset=run_args.dataset, mode='test', tokenizer=tokenizer)

        # Use Huggingface's Trainer class for training and evaluation
        training_args = Seq2SeqTrainingArguments(
            output_dir="test-bart-with-template",
            # evaluation_strategy="epoch",
            dataloader_num_workers=4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=4,
            num_train_epochs=2,
            load_best_model_at_end=False,
            predict_with_generate=True
            # eval_accumulation_steps=4
        )

        trainer = Seq2SeqTrainer(
            model,
            training_args,
            train_dataset=train_dataset,
            # eval_dataset=val_dataset,
            # compute_metrics=intent_metrics_bart,
            tokenizer=tokenizer
        )

        # trainer.train()
        # trainer.evaluate()
        # predict_results = model.generate(test_dataset[0]['input_ids'].unsqueeze(0),
        #   decoder_start_token_id=tokenizer.eos_token_id)
        # predict_results = trainer.predict(test_dataset)

        # Select a subset of the test dataset for evaluation because the whole set does not fit into GPU memory
        begin_prediction_range = 0
        num_predictions = 80

        to_predict = [test_dataset[i] for i in range(begin_prediction_range, begin_prediction_range + num_predictions)]
        predict_results = trainer.predict(to_predict, max_length=128)

        predictions = tokenizer.batch_decode(
            predict_results.predictions,
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

        for i in range(begin_prediction_range, begin_prediction_range + num_predictions):
            utter = test_dataset.sentences[i]
            tags = test_dataset.slots[i]

            gold_slots = SlotDataset.get_clean_slots_dict(utter, tags)
            pred_slots = convert_bart_output_to_slot_preds(predictions[i - begin_prediction_range])

            # For error inspecting
            # intent = test_dataset.intents[i]
            # intent_slot_mapping = test_dataset.intent_slots_mapping[intent]
            # if list(intent_slot_mapping) != list(pred_slots.keys()):
            #     print(intent_slot_mapping, '\n', pred_slots.keys())


            intent = test_dataset.intents[i]
            intent_slot_mapping = test_dataset.intent_slots_mapping[intent]
            for slot_name, slot_pred in zip(intent_slot_mapping, pred_slots):

            # for slot_name, slot_pred in pred_slots.items():

                slot_pred = slot_pred.lower()
                gold_slot = gold_slots.get(slot_name, '')
                if slot_pred != 'none':
                    if gold_slot == '':
                        scores[slot_name]["false_positives"] += 1
                        # print(f"Test example {i+1}\t{slot_name=}\t{slot_pred=}")
                    else:
                        if gold_slot == slot_pred:
                            # print(f"TPTest example {i + 1}\t{slot_name=}\t{slot_pred=}\t{gold_slot=}")
                            scores[slot_name]["true_positives"] += 1
                        else:
                            print(f"Test example {i + 1}\t{slot_name=}\t{slot_pred=}\t{gold_slot=}")
                            # Maybe skip the following?
                            scores[slot_name]["false_negatives"] += 1
                else:
                    if gold_slot != '':
                        # print(f"Test example {i + 1}\t{slot_name=}\t{slot_pred=}\t{gold_slot=}")
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

    elif run_args.approach.lower() == 'manual_prompting' and run_args.model_type.lower() == 'test-t5':
        # checkpoint = 'facebook/bart-base'
        # config = BartConfig.from_pretrained(checkpoint)
        # config.force_bos_token_to_be_generated = True
        # # config.forced_bos_token_id = tokenizer.bos_token_id
        #
        # tokenizer = BartTokenizer.from_pretrained(checkpoint)
        # model = BartForConditionalGeneration.from_pretrained(checkpoint, config=config)

        checkpoint = 't5-base'  # 't5-small'
        tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        model = T5ForConditionalGeneration.from_pretrained(checkpoint)

        train_dataset = SlotDataset(dataset=run_args.dataset, mode='train', tokenizer=tokenizer)
        val_dataset = SlotDataset(dataset=run_args.dataset, mode='dev', tokenizer=tokenizer)
        test_dataset = SlotDataset(dataset=run_args.dataset, mode='test', tokenizer=tokenizer)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, num_workers=4)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=4, num_workers=4)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=4, num_workers=4)

        train_seq2seq_model(model, tokenizer, train_dataloader, val_dataloader, test_dataloader, model_weights_path=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="snips", type=str, help="The input dataset")
    parser.add_argument("--approach", default="manual_prompting", type=str,
                        help="Select approach between `prompting` and `fine-tuning`")
    # parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--model_type", default="test-t5", type=str, help="Select model type")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    parser.add_argument("--do_train", default=True, type=bool, help="Whether to train the model.")
    parser.add_argument("--do_eval", default=True, type=bool, help="Whether to evaluate the model on the test set.")

    parser.add_argument("--save_preds", default=True, type=bool, help="Whether to save model's predictions to csv.")

    run_args = parser.parse_args()
    model_config = OmegaConf.load("config/model_config.yaml")

    main(run_args, model_config)
