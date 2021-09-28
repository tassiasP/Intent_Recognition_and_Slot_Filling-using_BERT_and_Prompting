import random
import numpy as np
import torch
import re
from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.scheme import IOB2
from sklearn.metrics import f1_score as sklearn_f1_score
from transformers import EvalPrediction
from datasets import load_metric


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slot_metrics(labels, preds):
    f1 = seqeval_f1_score(y_true=labels, y_pred=preds, mode='strict', scheme=IOB2)

    return f1


def intent_metrics(labels, preds):
    labels, preds = np.array(labels), np.array(preds)
    accuracy = (labels == preds).mean()
    f1 = sklearn_f1_score(y_true=labels, y_pred=preds, average='weighted')

    return accuracy, f1


def intent_metrics_bart(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    # Replace -100 (CrossEntropy's ignore_index) in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, 1)  # 1 is the pad_token_id
    # For error inspection
    # print(f"{preds=}\n\n {labels=}")

    metric = load_metric("accuracy")

    return metric.compute(predictions=preds.flatten(), references=labels.flatten())


def convert_output_to_slot_preds(prediction_output):
    # example:
    #   input = "<s> The playlist owner is none</s>The playlist is grime instrumentals</s> \
    #   The artist is sabrina salerno</s>The track name is none</s>The music item is none</s> \
    #   The</s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s>"
    #
    #   output: {"playlist owner": "none", "playlist": "grime instrumentals", "artist": "sabrina salerno",
    #             "track name": "none", "music item": none}
    slot_preds = {}
    for sent in prediction_output.split('</s>'):
        match = re.search(r"(?<=The).*", sent)
        if match:
            match = match.group(0).strip()
            # print(match)
            if match != '':
                slot, _, value = match.partition(' is ')
                slot, value = slot.strip(), value.strip()

                try:
                    slot_preds[INVERTED_SLOT_MAPPING[slot]] = value
                except KeyError:
                    continue

    return slot_preds


def compute_micro_f1(scores: dict):
    tps = 0  # true positives
    fps = 0  # false positives
    fns = 0  # false negatives
    for slot_scores in scores.values():
        tps += slot_scores["true_positives"]
        fps += slot_scores["false_positives"]
        fns += slot_scores["false_negatives"]

    print(f"{tps=}\t{fps=}\t{fns=}")
    micro_precision = tps / (tps + fps)
    micro_recall = tps / (tps + fns)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    return micro_precision, micro_recall, micro_f1


# To be used for BART mask filling
INTENT_MAPPING = {
    "UNK": "unknown",
    "AddToPlaylist": "playlist",
    "BookRestaurant": "restaurant",
    "GetWeather": "weather",
    "PlayMusic": "music",
    "RateBook": "book",
    "SearchCreativeWork": "creative work",
    "SearchScreeningEvent": "movie"
}

SLOT_MAPPING = {
    "UNK": "unknown",
    "album": "album",
    "artist": "artist",
    "best_rating": "best rating",
    "city": "city",
    "condition_description": "weather condition",
    "condition_temperature": "temperature condition",
    "country": "country",
    "cuisine": "cuisine",
    "current_location": "current location",
    "entity_name": "track name",
    "facility": "facility",
    "genre": "genre",
    "geographic_poi": "geographic area",
    "location_name": "location name",
    "movie_name": "movie name",
    "movie_type": "movie type",
    "music_item": "music item",
    "object_location_type": "location type",
    "object_name": "object name",
    "object_part_of_series_type": "type of series",
    "object_select": "selected object",
    "object_type": "object type",
    "party_size_description": "people description",
    "party_size_number": "number of people",
    "playlist": "playlist",
    "playlist_owner": "playlist owner",
    "poi": "point of interest",
    "rating_unit": "rating unit",
    "rating_value": "rating value",
    "restaurant_name": "name of the restaurant",
    "restaurant_type": "type of restaurant",
    "served_dish": "dish",
    "service": "service",
    "sort": "sort",
    "spatial_relation": "relative place",  # approximate area
    "state": "state",
    "timeRange": "time range",
    "track": "track to play",
    "year": "year"
}

INVERTED_SLOT_MAPPING = {
    v: k for k, v in SLOT_MAPPING.items()
}
