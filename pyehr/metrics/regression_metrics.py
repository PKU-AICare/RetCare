from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

# get regression metrics: mse, mae, rmse, r2

def get_regression_metrics(preds, labels):
    mse = MeanSquaredError(squared=True)
    rmse = MeanSquaredError(squared=False)
    mae = MeanAbsoluteError()
    r2 = R2Score()

    mse(preds, labels)
    rmse(preds, labels)
    mae(preds, labels)
    r2(preds, labels)

    # return a dictionary
    return {
        "mse": mse.compute().item(),
        "rmse": rmse.compute().item(),
        "mae": mae.compute().item(),
        "r2": r2.compute().item(),
    }