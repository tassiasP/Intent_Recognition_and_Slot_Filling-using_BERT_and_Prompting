import argparse

import torch.cuda
from omegaconf import OmegaConf
from transformers import TrainingArguments, Trainer, BartForConditionalGeneration, BartTokenizer

from joint_dataset import get_dataloader, IntentDataset
from models import JointBert
from train import train
from utils import set_seed, intent_metrics_bart, INTENT_MAPPING
from reader import Reader


def main(run_args, model_config):
    torch.cuda.empty_cache()
    set_seed(run_args.seed)

    reader = Reader(run_args.dataset)
    reader.construct_intent_and_slot_vocabs(write_to_disk=True)

    intent_labels, slot_labels = reader.get_intent_labels(), reader.get_slot_labels()

    if run_args.model_type.lower() == 'bert':
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

    elif run_args.model_type.lower() == 'bart':
        checkpoint = 'facebook/bart-base'
        tokenizer = BartTokenizer.from_pretrained(checkpoint)
        model = BartForConditionalGeneration.from_pretrained(checkpoint, forced_bos_token_id=tokenizer.bos_token_id)

        train_dataset = IntentDataset(dataset=run_args.dataset, mode='train', tokenizer=tokenizer)
        val_dataset = IntentDataset(dataset=run_args.dataset, mode='dev', tokenizer=tokenizer)
        test_dataset = IntentDataset(dataset=run_args.dataset, mode='test', tokenizer=tokenizer)

        training_args = TrainingArguments(
            "test-bart",
            evaluation_strategy="epoch",
            dataloader_num_workers=4,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            num_train_epochs=2
        )

        trainer = Trainer(
            model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=intent_metrics_bart
        )

        trainer.train()
        trainer.evaluate()

        predict_results = trainer.predict(test_dataset)
        predictions = tokenizer.batch_decode(
            predict_results.predictions[0].argmax(axis=-1),
            skip_special_tokens=True
        )

        gold_intents = [INTENT_MAPPING[intent] for intent in test_dataset.intents]
        correct_preds = sum(1 for pred, label in zip(predictions, gold_intents) if pred == label)
        accuracy = correct_preds / len(test_dataset)
        print(f"\n{accuracy = }")

        # Error analysis
        for pred, label, init_label in zip(predictions, gold_intents, test_dataset.intents):
            if pred != label:
                print(f"True label: {init_label: >25} \tPrediction: {pred}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--approach", default="fine-tune", type=str, help="Select approach between prompt"
                                                                          "and fine-tune")
    # parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    # parser.add_argument("--data_dir", default="./data", type=str, help="The input data directory")
    parser.add_argument("--dataset", default="snips", type=str, help="The input dataset")
    parser.add_argument("--model_type", default="bert", type=str, help="Select model type")
    parser.add_argument('--seed', type=int, default=42, help="Seed for reproducibility")

    parser.add_argument('--do_train', default=True, type=bool, help="Whether to train the model.")
    parser.add_argument("--do_eval", default=True, type=bool, help="Whether to evaluate the model on the test set.")

    parser.add_argument("--save_preds", default=True, type=bool, help="Whether to save model's predictions to csv.")

    run_args = parser.parse_args()
    model_config = OmegaConf.load("config/model_config.yaml")

    main(run_args, model_config)
