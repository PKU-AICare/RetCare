import os
from tqdm import tqdm

import pandas as pd

from utils.healthcare_context_utils import generate_prompt
from retcare import RetCare
from utils.keywords_utils import generate_keywords

dataset = "ckd"
data_url = f"ehr_datasets/{dataset}/processed/fold_1"
retriever_name = "MedCPT"
# corpus_name = "PubMed"
corpus_name = "Textbooks"
llm_model = "deepseek-chat"
retcare = RetCare(llm_name=llm_model, ensemble='select', retriever_name=retriever_name, corpus_name=corpus_name)
pids = pd.read_pickle(f'{data_url}/test_pid.pkl')


for patient_index, patient_id in tqdm(enumerate(pids), total=len(pids), desc=f"Processing patients in {dataset} dataset"):
    try:
        subcontext, hcontext = generate_prompt(dataset, data_url, patient_index, patient_id)
    except Exception as e:
        print(f"Patient {patient_id} failed in generating prompt with error: {e}")
        continue
    
    save_dir=f'./response/{dataset}/pid{patient_id}'
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "hcontext.txt"), 'w') as f:
        f.write(hcontext)

    try:
        keywords = generate_keywords(llm_model, subcontext)
        keywords = ', '.join(keywords['keywords'])
    except Exception as e:
        print(f"Patient {patient_id} failed in generating keywords with error: {e}")
        continue
    
    with open(os.path.join(save_dir, "keywords.txt"), 'w') as f:
        f.write(keywords)

    try:
        answer, snippets, scores, messages = retcare.answer(hcontext=hcontext, keywords=keywords, k=16, save_dir=save_dir)
    except Exception as e:
        print(f"Patient {patient_id} failed in answering with error: {e}")
        continue