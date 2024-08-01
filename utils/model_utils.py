from typing import Dict
import os
import random
import requests

import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# from .concare_new import vanilla_transformer_encoder
from .ckd_info import disease_english, original_disease


# RANDOM_SEED = 12345
# np.random.seed(RANDOM_SEED)
# random.seed(RANDOM_SEED)

# ckpt_url = "/home/wangzixiang/xiaoya-backend/media/checkpoints/ConCare/concare_ckd"
# checkpoint_concare = torch.load(ckpt_url, map_location=lambda storage, loc: storage)

# device = torch.device("cuda:0" if torch.cuda.is_available() == True else "cpu")
# model_concare = vanilla_transformer_encoder(device=device).to(device)
# optimizer_concare = torch.optim.Adam(model_concare.parameters(), lr=0.001)

# model_concare.load_state_dict(checkpoint_concare["net"])
# optimizer_concare.load_state_dict(checkpoint_concare["optimizer"])
# model_concare.eval()


# def run_concare(x: torch.Tensor):
#     time_step = x.size()[0]
#     if time_step == 1:
#         test_x = torch.stack((x, x), axis=0)
#         test_x = test_x.to(device=device, dtype=torch.float32)
#         test_len = np.array([test_x.size()[1], test_x.size()[1]])
#         test_len = torch.tensor(test_len, dtype=torch.int64)
#         test_output, context, attn = model_concare(test_x, test_len)
#         output = test_output.cpu().detach().reshape(2, 1).numpy()
#         context = context.cpu().detach().reshape(2, 1, -1).numpy()
#         attn = attn.cpu().detach().reshape(2, 1, -1).numpy()
#         return output[0], context[0], attn[0]
#     else:
#         test_x = []
#         test_len = []
#         for i in range(time_step):
#             cur_x = np.zeros((time_step, 17))
#             idx = time_step - i
#             cur_x[:idx] = x[:idx]
#             test_x.append(cur_x)
#             test_len.append(idx)
#         test_x = np.array(test_x)
#         test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
#         test_len = np.array(test_len)
#         test_len = torch.tensor(test_len, dtype=torch.int64)
#         test_output, context, attn = model_concare(test_x, test_len)
#         output = test_output.cpu().detach().numpy().squeeze()
#         output = (np.flip(output, axis=0) / 0.41)
#         context = context.cpu().detach().numpy().squeeze()
#         attn = attn.cpu().detach().numpy()
#         attn = np.flip(attn, axis=0)
#         return output, context, attn


# def run_ml_models(config, x):
#     checkpoint_path = f'logs/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}/checkpoints/best.ckpt'
#     model = pd.read_pickle(checkpoint_path)
#     output = model.predict(x)
#     feature_importance = model.get_feature_importance('shap')
#     return output, None, feature_importance


def get_data_from_files(data_url: str, model: str, patient_index: int):
    x = pd.read_pickle(os.path.join(data_url, "test_x.pkl"))[patient_index]
    raw_x = pd.read_pickle(os.path.join(data_url, "test_raw_x.pkl"))[patient_index]
    features = pd.read_pickle(os.path.join(data_url, "labtest_features.pkl"))
    
    y = pd.read_pickle(os.path.join(data_url, f"{model}_output.pkl"))[patient_index]
    important_features = pd.read_pickle(os.path.join(data_url, f"{model}_features.pkl"))[patient_index]
    return x, raw_x, features, y, important_features


# def get_similar_patients(data_url: str, patient_id: int) -> Dict:
#     patients_id = pd.read_pickle(os.path.join(data_url, 'pid.pkl'))
#     patients_x = pd.read_pickle(os.path.join(data_url, 'x.pkl'))
#     patients_context = []
#     for pid, x in zip(patients_id, patients_x):
#         if pid == patient_id:
#             patient_context = run_concare(torch.Tensor(x))[1][-1:]
#         else:
#             context = run_concare(torch.Tensor(x))[1]
#             patients_context.append(context[-1])
#     patients_id.remove(patient_id)
#     # similarity
#     patients_context = np.array(patients_context)
#     cosine = (np.dot(patient_context, patients_context.T) + 1e-10) / ((np.linalg.norm(patient_context, axis=1) + 1e-10) * (np.linalg.norm(patients_context, axis=1) + 1e-10))
#     similarity = cosine.squeeze()
    
#     data = []
#     for pid in patients_id:
#         basic_data = requests.get(f"http://47.93.42.104:10408/v1/app/patients/basics/{patient_id}").json()["data"]
#         gender = "male" if basic_data["gender"] == 1 else "female"
#         age = basic_data["age"]
#         ori_disease = original_disease[basic_data["originDisease"]]
#         basic_disease = [disease_english[key] for key in disease_english.keys() if basic_data[key] == 1]
#         basic_disease = ", and basic disease " + ", ".join(basic_disease) if len(basic_disease) > 0 else ""
#         data.append({
#             "pid": pid,
#             "similarity": similarity[patients_id.index(pid)],
#             "gender": gender,
#             "age": age,
#             "oriDisease": ori_disease,
#             "basicDisease": basic_disease,
#         })
#     data = sorted(data, key=lambda x: x['similarity'], reverse=True)
#     return data[:6]