import random
import numpy as np
import torch
from seqeval.metrics import f1_score as seqeval_f1_score
from sklearn.metrics import f1_score as sklearn_f1_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slot_metrics(labels, preds):
    f1 = seqeval_f1_score(y_true=labels, y_pred=preds)

    return f1


def intent_metrics(labels, preds):
    accuracy = (labels == preds).mean()
    f1 = sklearn_f1_score(y_true=labels, y_pred=preds)

    return accuracy, f1


def sentence_metrics():
    pass
