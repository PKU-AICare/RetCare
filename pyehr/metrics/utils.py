def check_metric_is_better(cur_best, main_metric, score, task):
    if task=="los":
        if cur_best=={}:
            return True
        if score < cur_best[main_metric]:
            return True
        return False
    elif task in ["outcome", "multitask"]:
        if cur_best=={}:
            return True
        if score > cur_best[main_metric]:
            return True
        return False
    else:
        raise ValueError("Task not supported")