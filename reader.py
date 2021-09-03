import os


class Reader:

    def __init__(self, dataset: str):
        self.data_path = './data'
        self.dataset = dataset
        self.text_file = 'seq.in'
        self.slot_file = 'seq.out'
        self.intent_file = 'label'

    @staticmethod
    def _read_file(filename, split=True):
        with open(filename, mode="r", encoding="utf-8") as f:
            return [line.strip().split() if split else line.strip() for line in f]

    def read_dataset(self, mode='train'):
        sentences = Reader._read_file(os.path.join(self.data_path, self.dataset, mode, self.text_file))
        slots = Reader._read_file(os.path.join(self.data_path, self.dataset, mode, self.slot_file))
        intents = Reader._read_file(os.path.join(self.data_path, self.dataset, mode, self.intent_file), split=False)
        assert len(sentences) == len(slots) == len(intents)

        return sentences, slots, intents

    def construct_intent_and_slot_vocabs(self, write_to_disk=True):
        _, slots, intents = self.read_dataset(mode='train')

        sorted_intent_labels = sorted(list(set(intents)))

        slot_labels = {slot for line in slots for slot in line}
        sorted_slot_labels = sorted(list(slot_labels),
                                    key=lambda slot_name: (slot_name[2:], slot_name[:2]))

        # TODO why unk?
        sorted_intent_labels = ["UNK"] + sorted_intent_labels
        sorted_slot_labels = ["UNK", "PAD"] + sorted_slot_labels

        if write_to_disk:
            with open(os.path.join(self.data_path, self.dataset, "intent_labels.txt"), mode="w", encoding="utf-8") as f:
                for intent in sorted_intent_labels:
                    f.write(intent + '\n')
            with open(os.path.join(self.data_path, self.dataset, "slot_labels.txt"), mode="w", encoding="utf-8") as f:
                for slot in sorted_slot_labels:
                    f.write(slot + '\n')
        else:
            return sorted_intent_labels, sorted_slot_labels

    def get_intent_labels(self):
        return Reader._read_file(os.path.join(self.data_path, self.dataset, "intent_labels.txt"), split=False)

    def get_slot_labels(self):
        return Reader._read_file(os.path.join(self.data_path, self.dataset, "slot_labels.txt"), split=False)
