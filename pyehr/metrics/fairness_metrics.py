import numpy as np

def calculate_di(y_pred, y_true, sensitive_dict, privileged_conditions):
    # 创建一个字典来存储每个敏感属性的DI值
    di_dict = {}

    # 对于每个敏感属性
    for attr, condition in privileged_conditions.items():
        # 提取这个敏感属性的值
        sensitive_values = np.array(sensitive_dict[attr])

        # 确定哪些样本属于特权群体和非特权群体
        condition_array = np.array([condition(val) for val in sensitive_values])
        privileged_indices = np.where(condition_array)[0]
        unprivileged_indices = np.where(~condition_array)[0]

        # 计算特权群体和非特权群体得到正向预测结果的概率
        privileged_positive_rate = np.mean(y_pred[privileged_indices] == 1)
        unprivileged_positive_rate = np.mean(y_pred[unprivileged_indices] == 1)

        # 计算并存储这个敏感属性的DI值
        if unprivileged_positive_rate / privileged_positive_rate<1:
            di_dict[attr] = unprivileged_positive_rate / privileged_positive_rate
        else:
            di_dict[attr] = privileged_positive_rate / unprivileged_positive_rate

    return di_dict


def calculate_aod(y_pred, y_true, sensitive_dict, privileged_conditions):
    aod_dict = {}

    for attr, condition in privileged_conditions.items():
        sensitive_values = np.array(sensitive_dict[attr])

        condition_array = np.array([condition(val) for val in sensitive_values])
        privileged_indices = np.where(condition_array)[0]
        unprivileged_indices = np.where(~condition_array)[0]

        y_pred_privileged = y_pred[privileged_indices]
        y_pred_unprivileged = y_pred[unprivileged_indices]
        y_true_privileged = y_true[privileged_indices]
        y_true_unprivileged = y_true[unprivileged_indices]

        def safe_divide(numerator, denominator):
            return numerator / denominator if denominator != 0 else 0  # or np.nan or some placeholder value

        TPR_privileged = safe_divide(np.sum((y_pred_privileged == 1) & (y_true_privileged == 1)), np.sum(y_true_privileged == 1))
        FPR_privileged = safe_divide(np.sum((y_pred_privileged == 1) & (y_true_privileged == 0)), np.sum(y_true_privileged == 0))
        TPR_unprivileged = safe_divide(np.sum((y_pred_unprivileged == 1) & (y_true_unprivileged == 1)), np.sum(y_true_unprivileged == 1))
        FPR_unprivileged = safe_divide(np.sum((y_pred_unprivileged == 1) & (y_true_unprivileged == 0)), np.sum(y_true_unprivileged == 0))

        aod_dict[attr] = 0.5 * ((FPR_privileged - FPR_unprivileged) + (TPR_privileged - TPR_unprivileged))

    return aod_dict



def calculate_eod(y_pred, y_true, sensitive_dict, privileged_conditions):
    eod_dict = {}

    for attr, condition in privileged_conditions.items():
        sensitive_values = np.array(sensitive_dict[attr])

        condition_array = np.array([condition(val) for val in sensitive_values])
        privileged_indices = np.where(condition_array)[0]
        unprivileged_indices = np.where(~condition_array)[0]

        y_pred_privileged = y_pred[privileged_indices]
        y_pred_unprivileged = y_pred[unprivileged_indices]
        y_true_privileged = y_true[privileged_indices]
        y_true_unprivileged = y_true[unprivileged_indices]

        def safe_divide(numerator, denominator):
            return numerator / denominator if denominator != 0 else 0  # or np.nan or some placeholder value

        TPR_privileged = safe_divide(np.sum((y_pred_privileged == 1) & (y_true_privileged == 1)), np.sum(y_true_privileged == 1))
        TPR_unprivileged = safe_divide(np.sum((y_pred_unprivileged == 1) & (y_true_unprivileged == 1)), np.sum(y_true_unprivileged == 1))

        eod_dict[attr] = TPR_privileged - TPR_unprivileged

    return eod_dict


def calculate_spd(y_pred, y_true, sensitive_dict, privileged_conditions):
    # 创建一个字典来存储每个敏感属性的SPD值
    spd_dict = {}

    # 对于每个敏感属性
    for attr, condition in privileged_conditions.items():
        # 提取这个敏感属性的值
        sensitive_values = np.array(sensitive_dict[attr])

        # 确定哪些样本属于特权群体和非特权群体
        condition_array = np.array([condition(val) for val in sensitive_values])
        privileged_indices = np.where(condition_array)[0]
        unprivileged_indices = np.where(~condition_array)[0]

        # 提取特权群体和非特权群体的预测结果
        y_pred_privileged = y_pred[privileged_indices]
        y_pred_unprivileged = y_pred[unprivileged_indices]

        # 计算特权群体和非特权群体的正例预测率
        PPR_privileged = np.mean(y_pred_privileged == 1)
        PPR_unprivileged = np.mean(y_pred_unprivileged == 1)

        # 计算并存储这个敏感属性的SPD值
        spd_dict[attr] = PPR_privileged - PPR_unprivileged

    return spd_dict



def calculate_bias(y_pred, y_true, sensitive_dict, privileged_conditions, threshold):
    y_true = np.array(y_true)
    y_pred_class = [1 if y >= threshold else 0 for y in y_pred]
    y_pred_class = np.array(y_pred_class)
    performance_dict = {}
    performance_dict['di'] = calculate_di(y_pred_class, y_true, sensitive_dict, privileged_conditions)
    performance_dict['aod'] = calculate_aod(y_pred_class, y_true, sensitive_dict, privileged_conditions)
    performance_dict['eod'] = calculate_eod(y_pred_class, y_true, sensitive_dict, privileged_conditions)
    performance_dict['spd'] = calculate_spd(y_pred_class, y_true, sensitive_dict, privileged_conditions)

    return performance_dict


