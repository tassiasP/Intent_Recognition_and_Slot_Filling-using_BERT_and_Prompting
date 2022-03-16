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
    f1 = seqeval_f1_score(y_true=labels, y_pred=preds, mode="strict", scheme=IOB2)

    return f1


def intent_metrics(labels, preds):
    labels, preds = np.array(labels), np.array(preds)
    accuracy = (labels == preds).mean()
    f1 = sklearn_f1_score(y_true=labels, y_pred=preds, average="weighted")

    return accuracy, f1


def convert_t5_output_to_slot_preds(pred):
    """Converts raw prediction into slot prediction using the T5 sentinel tokens (<extra_id_0>, <extra_id_1> etc.)"""
    match = re.split(r"<extra_id_\d+>", pred)
    if match:
        # Skip the <pad> and </s> (eos) tokens at the start and the end of the output respectively
        return match[1:-1]


def compute_micro_f1(scores: dict):
    tps = 0  # true positives
    fps = 0  # false positives
    fns = 0  # false negatives
    for slot_scores in scores.values():
        tps += slot_scores["true_positives"]
        fps += slot_scores["false_positives"]
        fns += slot_scores["false_negatives"]

    print(
        f"# of True Positives= {tps}\t# of False Positives= {fps}\t# of False Negatives= {fns}"
    )
    micro_precision = tps / (tps + fps)
    micro_recall = tps / (tps + fns)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    return micro_precision, micro_recall, micro_f1


# To be used for mask filling
INTENT_MAPPING = {
    "UNK": "unknown",
    "AddToPlaylist": "playlist",
    "BookRestaurant": "restaurant",
    "GetWeather": "weather",
    "PlayMusic": "music",
    "RateBook": "book",
    "SearchCreativeWork": "creative work",
    "SearchScreeningEvent": "movie",
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
    "year": "year",
}

INVERTED_SLOT_MAPPING = {v: k for k, v in SLOT_MAPPING.items()}
