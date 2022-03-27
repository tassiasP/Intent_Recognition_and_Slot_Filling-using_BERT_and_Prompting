import argparse

from omegaconf import OmegaConf
import torch.cuda
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, T5Config, T5Tokenizer, T5ForConditionalGeneration
from transformers import logging

from joint_dataset import get_dataloader, IntentDataset, SlotDataset
from models import JointBert, T5PromptTuningLM
from train_evaluate import train, train_seq2seq_model
from utils import set_seed, INTENT_MAPPING, ATIS_INTENT_MAPPING
from reader import Reader


def main(run_args, model_config):
    torch.cuda.empty_cache()
    set_seed(run_args.seed)

    reader = Reader(run_args.dataset)
    reader.construct_intent_and_slot_vocabs(write_to_disk=True)

    intent_labels, slot_labels = reader.get_intent_labels(), reader.get_slot_labels()

    if run_args.approach.lower() == 'fine-tuning':
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

    elif run_args.approach == 'prompting':
        if run_args.predict_intent:
            checkpoint = 't5-small'
            tokenizer = T5Tokenizer.from_pretrained(checkpoint)
            # checkpoint = './test-t5-intent/checkpoint-2000'
            model = T5ForConditionalGeneration.from_pretrained(checkpoint)

            use_prompting = True

            train_dataset = IntentDataset(dataset=run_args.dataset, mode='train', tokenizer=tokenizer,
                                          use_prompting=use_prompting)
            val_dataset = IntentDataset(dataset=run_args.dataset, mode='dev', tokenizer=tokenizer,
                                        use_prompting=use_prompting)
            test_dataset = IntentDataset(dataset=run_args.dataset, mode='test', tokenizer=tokenizer,
                                         use_prompting=use_prompting)

            # Try out a subset of the training set (few-shot)
            indices = torch.randperm(len(train_dataset))[:200]
            train_dataset = Subset(train_dataset, indices=indices)

            # Use Huggingface's Trainer class for training and evaluation
            training_args = Seq2SeqTrainingArguments(
                output_dir="test-t5-intent",
                evaluation_strategy="steps",
                save_strategy="steps",
                dataloader_num_workers=4,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
                num_train_epochs=18,
                logging_steps=50,
                learning_rate=1e-3,
                load_best_model_at_end=True,
                greater_is_better=False,
                predict_with_generate=True  # important in order to prevent teacher-forcing during inference
            )

            trainer = Seq2SeqTrainer(
                model,
                training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer
            )

            trainer.train()

            predict_results = trainer.predict(test_dataset)
            predictions = tokenizer.batch_decode(predict_results.predictions, skip_special_tokens=True)

            # Evaluation
            gold_intents = [INTENT_MAPPING[intent] if run_args.dataset == 'snips' else ATIS_INTENT_MAPPING[intent]
                            for intent in test_dataset.intents]
            correct_preds = sum(1 for pred, label in zip(predictions, gold_intents) if pred.strip() == label)
            accuracy = correct_preds / len(test_dataset)
            print(f"\n{accuracy = :.3f}\n")

            # Error analysis
            for pred, label, init_label in zip(predictions, gold_intents, test_dataset.intents):
                if pred.strip() != label:
                    print(f"True label: {init_label: >15} \tMapped label: {label: >15}\tPrediction: {pred}")

        elif run_args.predict_slots:
            logging.set_verbosity(logging.CRITICAL)
            use_prompting = True

            checkpoint = 't5-small'  # alternatively use 't5-base'
            tokenizer = T5Tokenizer.from_pretrained(checkpoint)
            model = T5ForConditionalGeneration.from_pretrained(checkpoint)

            # As an extra experiment, try to freeze decoder's parameters
            # for param in model.get_decoder().parameters():
            #     param.requires_grad = False

            print(f"Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

            # Test soft-prompts
            # model = T5PromptTuningLM.from_pretrained(checkpoint)
            # print(f"Number of trainable parameters: {model.get_num_trainable_params()}")

            train_dataset = SlotDataset(dataset=run_args.dataset, mode='train', tokenizer=tokenizer,
                                        use_prompting=use_prompting)
            val_dataset = SlotDataset(dataset=run_args.dataset, mode='dev', tokenizer=tokenizer,
                                      use_prompting=use_prompting)
            test_dataset = SlotDataset(dataset=run_args.dataset, mode='test', tokenizer=tokenizer,
                                       use_prompting=use_prompting)

            # Try out a subset of the training set (few-shot)
            indices = torch.randperm(len(train_dataset))[:200]
            train_dataset = Subset(train_dataset, indices=indices)

            # Few-shot for 2 intents
            # few_shot_labels = ['RateBook']
            # # few_shot_labels = ['PlayMusic', 'GetWeather']
            # few_shot_labels = ['AddToPlaylist', 'BookRestaurant']
            # full_indices = [i for i in range(len(train_dataset)) if train_dataset.intents[i] not in few_shot_labels]
            #
            # few_shot_indices_1 = torch.tensor(
            #     [i for i in range(len(train_dataset)) if train_dataset.intents[i] in few_shot_labels[0]])
            # few_shot_indices_2 = torch.tensor(
            #     [i for i in range(len(train_dataset)) if train_dataset.intents[i] in few_shot_labels[1]])
            #
            # num_of_examples_per_intent = 20
            # indices_to_keep_1 = torch.randperm(len(few_shot_indices_1))[:num_of_examples_per_intent]
            # indices_to_keep_2 = torch.randperm(len(few_shot_indices_2))[:num_of_examples_per_intent]
            #
            # few_shot_indices_1 = few_shot_indices_1[indices_to_keep_1].tolist()
            # few_shot_indices_2 = few_shot_indices_2[indices_to_keep_2].tolist()
            #
            # final_train_indices = [*full_indices, *few_shot_indices_1, *few_shot_indices_2]
            # # final_train_indices = [*full_indices, *few_shot_indices_1]
            # train_dataset = Subset(train_dataset, final_train_indices)
            #
            # val_indices = [i for i in range(len(val_dataset)) if val_dataset.intents[i] in few_shot_labels]
            # test_indices = [i for i in range(len(test_dataset)) if test_dataset.intents[i] in few_shot_labels]
            # val_dataset = Subset(val_dataset, val_indices)
            # test_dataset = Subset(test_dataset, test_indices)

            batch_size = 8
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
            val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4)
            test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=4)

            train_seq2seq_model(model, tokenizer, train_dataloader, val_dataloader, test_dataloader)
    else:
        print("This approach is not supported, choose between `fine-tuning` and `prompting`")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="atis", type=str, help="The input dataset")
    parser.add_argument("--approach", default="prompting", type=str,
                        help="Select approach between `prompting` and `fine-tuning`")
    parser.add_argument("--predict_intent", default=False, type=bool, help="Select whether to predict the intent")
    parser.add_argument("--predict_slots", default=True, type=bool, help="Select whether to predict the slots")

    # parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--model_type", default="t5", type=str, help="Select model type")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    parser.add_argument("--do_train", default=True, type=bool, help="Whether to train the model.")
    parser.add_argument("--do_eval", default=True, type=bool, help="Whether to evaluate the model on the test set.")

    parser.add_argument("--save_preds", default=True, type=bool, help="Whether to save model's predictions to csv.")

    run_args = parser.parse_args()
    model_config = OmegaConf.load("config/model_config.yaml")

    main(run_args, model_config)
