import argparse

import torch.cuda
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from omegaconf import OmegaConf
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    T5Config,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from transformers import logging

from joint_dataset import get_dataloader, IntentDataset, SlotDataset
from models import JointBert
from train_evaluate import train, train_seq2seq_model
from utils import set_seed, INTENT_MAPPING #intent_metrics_bart
from reader import Reader
import pdb


def main(run_args, model_config):
    pdb.set_trace()
    torch.cuda.empty_cache()
    set_seed(run_args.seed)
    # logging.set_verbosity(logging.CRITICAL)

    reader = Reader(run_args.dataset)
    reader.construct_intent_and_slot_vocabs(write_to_disk=True)

    intent_labels, slot_labels = reader.get_intent_labels(), reader.get_slot_labels()

    if run_args.approach.lower() == "fine-tuning":
        if run_args.model_type.lower() == "bert":
            model = JointBert(model_config, len(intent_labels), len(slot_labels))

            if run_args.do_train and run_args.do_eval:
                train_dataloader = get_dataloader(
                    run_args.dataset,
                    mode="train",
                    batch_size=model_config.batch_size,
                    model_name=model_config.model,
                )
                val_dataloader = get_dataloader(
                    run_args.dataset,
                    mode="dev",
                    batch_size=model_config.batch_size,
                    model_name=model_config.model,
                )
                test_dataloader = get_dataloader(
                    run_args.dataset,
                    mode="test",
                    batch_size=model_config.batch_size,
                    model_name=model_config.model,
                )
                if run_args.save_preds:
                    intent_preds, slot_preds = train(
                        model,
                        train_dataloader,
                        val_dataloader,
                        test_dataloader,
                        model_config,
                        intent_labels,
                        slot_labels,
                    )
                    reader.save_test_preds_to_csv(intent_preds, slot_preds)

                else:
                    train(
                        model,
                        train_dataloader,
                        val_dataloader,
                        test_dataloader,
                        model_config,
                        intent_labels,
                        slot_labels,
                    )

    elif run_args.approach == "prompting":
        if run_args.predict_intent:
            checkpoint = "t5-small"
            tokenizer = T5Tokenizer.from_pretrained(checkpoint)
            # checkpoint = './test-t5-intent/checkpoint-2000'
            model = T5ForConditionalGeneration.from_pretrained(checkpoint)

            train_dataset = IntentDataset(
                dataset=run_args.dataset, mode="train", tokenizer=tokenizer
            )
            val_dataset = IntentDataset(
                dataset=run_args.dataset, mode="dev", tokenizer=tokenizer
            )
            test_dataset = IntentDataset(
                dataset=run_args.dataset, mode="test", tokenizer=tokenizer
            )

            # Use Huggingface's Trainer class for training and evaluation
            training_args = Seq2SeqTrainingArguments(
                output_dir="test-t5-intent",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                dataloader_num_workers=4,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
                num_train_epochs=6,
                load_best_model_at_end=True,
                predict_with_generate=True,  # important in order to prevent teacher-forcing during inference
            )

            trainer = Seq2SeqTrainer(
                model,
                training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
            )

            trainer.train()

            predict_results = trainer.predict(test_dataset)
            predictions = tokenizer.batch_decode(
                predict_results.predictions, skip_special_tokens=True
            )

            # Evaluation
            gold_intents = [INTENT_MAPPING[intent] for intent in test_dataset.intents]
            correct_preds = sum(
                1
                for pred, label in zip(predictions, gold_intents)
                if pred.strip() == label
            )
            accuracy = correct_preds / len(test_dataset)
            print(f"\n{accuracy = :.3f}\n")

            # Error analysis
            for pred, label, init_label in zip(
                predictions, gold_intents, test_dataset.intents
            ):
                if pred.strip() != label:
                    print(
                        f"True label: {init_label: >15} \tMapped label: {label: >15}\tPrediction: {pred}"
                    )

        elif run_args.predict_slots:
            checkpoint = "t5-small"  # alternatively use t5-base
            tokenizer = T5Tokenizer.from_pretrained(checkpoint)
            model = T5ForConditionalGeneration.from_pretrained(checkpoint)

            train_dataset = SlotDataset(
                dataset=run_args.dataset,
                mode="train",
                tokenizer=tokenizer,
                use_prompting=True,
            )
            val_dataset = SlotDataset(
                dataset=run_args.dataset,
                mode="dev",
                tokenizer=tokenizer,
                use_prompting=True,
            )
            test_dataset = SlotDataset(
                dataset=run_args.dataset,
                mode="test",
                tokenizer=tokenizer,
                use_prompting=True,
            )

            train_dataloader = DataLoader(
                train_dataset, shuffle=True, batch_size=8, num_workers=4
            )
            val_dataloader = DataLoader(
                val_dataset, shuffle=False, batch_size=8, num_workers=4
            )
            test_dataloader = DataLoader(
                test_dataset, shuffle=False, batch_size=8, num_workers=4
            )

            train_seq2seq_model(
                model,
                tokenizer,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                model_weights_path=None,
            )
    else:
        print(
            "This approach is not supported, choose between `fine-tuning` and `prompting`"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", default="snips", type=str, help="The input dataset"
    )
    parser.add_argument(
        "--approach",
        default="prompting",
        type=str,
        help="Select approach between `prompting` and `fine-tuning`",
    )
    parser.add_argument(
        "--predict_intent",
        default=False,
        type=bool,
        help="Select whether to predict the intent",
    )
    parser.add_argument(
        "--predict_slots",
        default=True,
        type=bool,
        help="Select whether to predict the slots",
    )

    # parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument(
        "--model_type", default="t5", type=str, help="Select model type"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    parser.add_argument(
        "--do_train", default=True, type=bool, help="Whether to train the model."
    )
    parser.add_argument(
        "--do_eval",
        default=True,
        type=bool,
        help="Whether to evaluate the model on the test set.",
    )

    parser.add_argument(
        "--save_preds",
        default=True,
        type=bool,
        help="Whether to save model's predictions to csv.",
    )

    run_args = parser.parse_args()
    model_config = OmegaConf.load("config/model_config.yaml")

    main(run_args, model_config)
