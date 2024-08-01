"""
This script calculates the Outcome-Specific MAE (OSMAE) score for predictions of patient outcomes and length of stay (LOS).

The script contains four functions:

1. calculate_epsilon: This function calculates the decay term epsilon used in the OSMAE score. It takes three arguments: the true LOS, a threshold value, and a large LOS value (typically the 95th percentile of LOS values). It returns the decay term epsilon.

2. calculate_osmae: This function calculates the OSMAE score for a single prediction. It takes five arguments: the predicted LOS, the true LOS, a large LOS value, a threshold value, and a case string ("true" if the outcome was predicted correctly, "false" otherwise). It returns the calculated OSMAE score.

3. calculate_outcome_prediction_result: This function determines whether a prediction of the outcome is correct or not. It takes the predicted and true outcomes as arguments and returns "true" if the prediction is correct (i.e., the predicted outcome is greater than 0.5 and the true outcome is 1, or the predicted outcome is less or equal to 0.5 and the true outcome is 0), and "false" otherwise.

4. osmae_score: This is the main function of the script. It takes the true and predicted outcomes and LOS values, the large LOS value, and the threshold value as arguments, and calculates the OSMAE score over all the provided records. It returns the mean OSMAE score.

The if __name__ == "__main__": block at the end of the script is used to test the osmae_score function with a single record.

This script assumes that all the inputs are numpy arrays with the same length (equal to the number of records), and that the large LOS value and the threshold value are scalars. The verbose argument in the osmae_score function can be used to print the OSMAE scores of individual records.

Note: This script does not handle missing values in the inputs. It also does not check the validity of the inputs (e.g., whether the outcomes are binary values, whether the LOS values are non-negative, etc.). These checks should be performed before calling the osmae_score function.
"""


import numpy as np


def calculate_outcome_prediction_result(outcome_pred, outcome_true):
    outcome_pred = 1 if outcome_pred > 0.5 else 0
    return "true" if outcome_pred == outcome_true else "false"


def calculate_epsilon(los_true, threshold, large_los):
    """
    epsilon is the decay term
    """
    if los_true <= threshold:
        return 1
    else:
        return max(0, (los_true - large_los) / (threshold - large_los))


def calculate_osmae(los_pred, los_true, large_los, threshold, case="true"):
    if case == "true":
        epsilon = calculate_epsilon(los_true, threshold, large_los)
        return epsilon * np.abs(los_pred - los_true)
    elif case == "false":
        epsilon = calculate_epsilon(los_true, threshold, large_los)
        return epsilon * (max(0, large_los - los_pred) + max(0, large_los - los_true))
    else:
        raise ValueError("case must be 'true' or 'false'")


def osmae_score(
    y_true_outcome,
    y_true_los,
    y_pred_outcome,
    y_pred_los,
    large_los,
    threshold,
    verbose=0,
):
    """
    Args:
        - large_los: 95% largest LOS value (patient-wise)
        - threshold: 50%*mean_los (patient-wise) 

    Note:
        - y/predictions are already flattened here
        - so we don't need to consider visits_length
    """
    metric = []
    num_records = len(y_pred_outcome)
    for i in range(num_records):
        cur_outcome_pred = y_pred_outcome[i]
        cur_los_pred = y_pred_los[i]
        cur_outcome_true = y_true_outcome[i]
        cur_los_true = y_true_los[i]
        prediction_result = calculate_outcome_prediction_result(
            cur_outcome_pred, cur_outcome_true
        )
        metric.append(
            calculate_osmae(
                cur_los_pred,
                cur_los_true,
                large_los,
                threshold,
                case=prediction_result,
            )
        )
    result = np.array(metric)
    if verbose:
        print("OSMAE Score:", result)
    return {"osmae": result.mean(axis=0).item()}

if __name__ == "__main__":
    y_true_outcome = np.array([1])
    y_true_los = np.array([60])
    y_pred_outcome = np.array([0.7])
    y_pred_los = np.array([10])
    large_los = 110
    threshold = 10
    print(osmae_score(
        y_true_outcome,
        y_true_los,
        y_pred_outcome,
        y_pred_los,
        large_los,
        threshold,
        verbose=0,
    ))