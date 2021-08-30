import argparse
from omegaconf import OmegaConf

from joint_dataset import get_dataloader
from models import JointBert
from train import train
from utils import set_seed
from reader import Reader


def main(run_args, model_config):
    set_seed(run_args.seed)

    reader = Reader(run_args.dataset)
    reader.read_dataset()
    reader.construct_intent_and_slot_label_mapping()

    intent_labels, slot_labels = reader.get_intent_labels(), reader.get_slot_labels()

    if run_args.model_type == 'bert':
        model = JointBert(model_config, len(intent_labels), len(slot_labels))

        val_dataloader = get_dataloader(run_args.dataset,
                                        mode='dev',
                                        batch_size=model_config.batch_size,
                                        model_name=model_config.model)
        test_dataloader = get_dataloader(run_args.dataset,
                                         mode='test',
                                         batch_size=model_config.batch_size,
                                         model_name=model_config.model)

        if run_args.do_train:
            train_dataloader = get_dataloader(run_args.dataset,
                                              mode='train',
                                              batch_size=model_config.batch_size,
                                              model_name=model_config.model)
            train(model, train_dataloader, model_config, intent_labels, slot_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--approach", default="fine-tune", type=str, help="Select approach between prompt"
                                                                          "and fine-tune")
    # parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    # parser.add_argument("--data_dir", default="./data", type=str, help="The input data directory")
    parser.add_argument("--dataset", default="atis", type=str, help="The input dataset")
    parser.add_argument("--model_type", default="bert", type=str, help="Select model type")
    parser.add_argument('--seed', type=int, default=42, help="Seed for reproducibility")

    parser.add_argument('--do_train', default=True, type=bool, help="Whether to train the model.")
    parser.add_argument("--do_eval", default=True, type=bool, help="Whether to evaluate the model on the test set.")

    run_args = parser.parse_args()
    model_config = OmegaConf.load("config/model_config.yaml")

    # processor = JointProcessor()
    # # toy test example
    # processed = processor.convert_example_to_bert_features("This is the transformative library",
    #                                                        12,
    #                                                        [0, 0, 0, 2, 5])
    #
    # print(processed)

    main(run_args, model_config)
