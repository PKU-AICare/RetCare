import json
import pdb
import os
import requests
from typing import List, Dict

import torch
import pandas as pd
import numpy as np

from .ckd_info import medical_standard, disease_english, medical_name, medical_unit, original_disease
from .model_utils import run_concare, run_ml_models, get_data_from_files, get_similar_patients


def get_var_desc(var: float):
    if var > 0:
        return round(var * 100, 2)
    else:
        return round(-var * 100, 2)


def get_trend_desc(var: float):
    if var > 0:
        return "increased"
    else:
        return "decreased"
    
    
def get_recommended_trend_desc(var: float):
    if var > 0:
        return "decrease"
    else:
        return "increase"
    
    
def get_range_desc(key: str, var: float):
    if key in ["Weight", "Appetite"]:
        return ""
    if var < medical_standard[key][0]:
        return f"the value is lower than normal range by {round((medical_standard[key][0] - var) / medical_standard[key][0] * 100, 2)}%"
    elif var > medical_standard[key][1]:
        return f"the value is higher than normal range by {round((var - medical_standard[key][1]) / medical_standard[key][1] * 100, 2)}%"
    else:
        return "the value is within the normal range"


def get_mean_desc(var: str, mean: float):
    if var < mean:
        return f"lower by {round((mean - var) / mean * 100, 0)}%"
    elif var > mean:
        return f"higher by {round((var - mean) / mean * 100, 0)}%"


def get_death_desc(risk: float):
    if risk < 0.5:
        return "a low level"
    elif risk < 0.7:
        return "a high level"
    else:
        return "an extremely high level"
    
    
def get_distribution(data, values):
    arr = np.sort(np.array(values))
    index = np.searchsorted(arr, data, side='right')
    rank = index / len(arr) * 100
    if rank < 40:
        return "at the bottom 40%"
    elif rank < 70:
        return "at the middle 30%"
    else:
        return "at the top 30%"


def format_input_ehr(raw_x: List[List[float]], features: List[str]):
    raw_x = np.array(raw_x)[:, 2:]
    ehr = ""
    for i, feature in enumerate(features):
        name = medical_name[feature] if feature in medical_name else feature
        ehr += f"- {name}: \"{', '.join(list(map(lambda x: str(round(x, 2)), raw_x[:, i])))}\"\n"
    return ehr


def generate_prompt(dataset: str, data_url: str, patient_index: int, patient_id: int):
    if dataset == 'ckd':
        basic_data = pd.read_pickle(os.path.join(data_url, 'basic.pkl'))[patient_id]
        gender = "male" if basic_data["Gender"] == 1 else "female"
        age = basic_data["Age"]
        if " " in basic_data["Origin_disease"]:
            ori_disease = basic_data["Origin_disease"].split(" ")[0]
            ori_disease = original_disease[ori_disease]
        else:
            ori_disease = original_disease[basic_data["Origin_disease"]]
        basic_disease = [disease_english[key] for key in disease_english.keys() if basic_data[key] == 1]
        basic_disease = ", and basic disease " + ", ".join(basic_disease) if len(basic_disease) > 0 else ""
        basic_context = f"This {gender} patient, aged {age}, is an End-Stage Renal Disease(ESRD) patient with original disease {ori_disease}{basic_disease}.\n"
    elif dataset == 'cdsl':
        basic_data = pd.read_pickle(os.path.join(data_url, 'basic.pkl'))[patient_id]
        gender = "male" if basic_data["Sex"] == 1 else "female"
        age = basic_data["Age"]
        basic_context = f"This {gender} patient, aged {age}, is an patient admitted with a diagnosis of COVID-19 or suspected COVID-19 infection.\n"
    else: # [mimic-iii, mimic-iv]
        basic_context = '\n'

    models = ['LR', 'ConCare']
    last_visit_context = f"We have {len(models)} models {', '.join(models)} to predict the mortality risk and estimate the feature importance weight for the patient in the last visit:\n"
    for model in models:
        _, raw_x, features, y, important_features = get_data_from_files(data_url, model, patient_index)
        ehr_context = "Here is complete medical information from multiple visits of a patient, with each feature within this data as a string of values separated by commas.\n" + format_input_ehr(raw_x, features)

        last_visit = f"The mortality prediction risk for the patient from {model} model is {round(float(y), 2)} out of 1.0, which means the patient is at {get_death_desc(float(y))} of death risk. Our model especially pays great attention to following features:\n"

        survival_stats = pd.read_pickle(os.path.join(data_url, 'survival.pkl'))
        dead_stats = pd.read_pickle(os.path.join(data_url, 'dead.pkl'))
        for item in important_features:
            key, value = item
            if key in ['Weight', 'Appetite']:
                continue
            survival_mean = survival_stats[key]['mean']
            dead_mean = dead_stats[key]['mean']
            key_name = medical_name[key] if key in medical_name else key
            key_unit = ' ' + medical_unit[key] if key in medical_unit else ''
            last_visit += f'{key_name}: with '
            if model == 'ConCare':
                last_visit += f'importance weight of {round(float(value["attention"]), 3)} out of 1.0. '
            else:
                last_visit += f'shap value of {round(float(value["attention"]), 3)}. '
            last_visit += f'The feature value is {round(value["value"], 2)}{key_unit}, which is {get_mean_desc(value["value"], survival_mean)} than the average value of survival patients ({round(survival_mean, 2)}{key_unit}), {get_mean_desc(value["value"], dead_mean)} than the average value of dead patients ({round(dead_mean, 2)}{key_unit}).\n'
        last_visit_context += last_visit + '\n'
    
    # similar_patients = get_similar_patients(data_url, patient_id)
    # similar_context = "The AI model has found similar patients to the patient, including:\n"
    # for idx, patient in enumerate(similar_patients):
    #     similar_context += f"Patient {idx + 1}: {patient['gender']}, {patient['age']} years old, with original disease {patient['oriDisease']}{patient['basicDisease']}{patient['deathText']}.\n"


    subcontext = basic_context + last_visit_context
    hcontext = basic_context + '\n' + ehr_context + '\n' + last_visit_context

    return subcontext, hcontext
        