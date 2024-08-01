import torch

from .binary_classification_metrics import get_binary_metrics
from .es import es_score
from .osmae import osmae_score
from .regression_metrics import get_regression_metrics
from .utils import check_metric_is_better


def reverse_los(y, los_info):
    return y * los_info["los_std"] + los_info["los_mean"]

def get_all_metrics(preds, labels, task, los_info):
    # threshold = los_info["threshold"]
    # large_los = los_info["large_los"]

    # convert preds and labels to tensor if they are ndarray type
    if isinstance(preds, torch.Tensor) == False:
        preds = torch.tensor(preds)
    if isinstance(labels, torch.Tensor) == False:
        labels = torch.tensor(labels)

    if task == "outcome":
        return get_binary_metrics(preds, labels[:, 0]) if len(labels.shape) > 1 else get_binary_metrics(preds, labels)
    elif task == "los":
        y_pred_los = reverse_los(preds, los_info)
        y_true_los = reverse_los(labels[:,1], los_info)
        return get_regression_metrics(y_pred_los, y_true_los)
    # elif task == "multitask":
    #     if not isinstance(threshold, list):
    #         y_pred_los = reverse_los(preds[:,1], los_info)
    #         y_true_los = reverse_los(labels[:,1], los_info)
    #         return get_binary_metrics(preds[:,0], labels[:,0]) | get_regression_metrics(y_pred_los, y_true_los) | osmae_score(labels[:,0], y_true_los, preds[:,0], y_pred_los, large_los, threshold) | es_score(labels[:,0], y_true_los, preds[:,0], threshold)
    #     else:
    #         y_pred_los = reverse_los(preds[:,1], los_info)
    #         y_true_los = reverse_los(labels[:,1], los_info)
    #         osmae_list = []
    #         es_list = []
    #         for thr in threshold:
    #             osmae_list.append(osmae_score(labels[:,0], y_true_los, preds[:,0], y_pred_los, large_los, thr)['osmae'])
    #             es_list.append(es_score(labels[:,0], y_true_los, preds[:,0], thr)['es'])
    #         return {"osmae_list": osmae_list, "es_list": es_list}
    else:
        raise ValueError("Task not supported")