import random
import numpy as np
import torch
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
    # For error inspection
    # print(f"{preds=}\n\n {labels=}")

    metric = load_metric("accuracy")

    return metric.compute(predictions=preds.flatten(), references=labels.flatten())


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

