import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision
from sklearn import metrics as sklearn_metrics
import numpy as np

threshold = 0.5

def minpse(preds, labels):
    precisions, recalls, thresholds = sklearn_metrics.precision_recall_curve(labels, preds)
    minpse_score = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    return minpse_score

def get_binary_metrics(preds, labels):

    accuracy = Accuracy(task="binary", threshold=threshold)
    auroc = AUROC(task="binary")
    auprc = AveragePrecision(task="binary")

    # convert labels type to int
    labels = labels.type(torch.int)
    accuracy(preds, labels)
    auroc(preds, labels)
    auprc(preds, labels)
    minpse_score = minpse(preds, labels)

    # return a dictionary
    return {
        "accuracy": accuracy.compute().item(),
        "auroc": auroc.compute().item(),
        "auprc": auprc.compute().item(),
        "minpse": minpse_score,
    }