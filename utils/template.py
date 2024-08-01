from liquid import Template

keywords_system = '''
You are a helpful medical expert with extensive medical knowledge. Your task is to searching for information about the context by calling the following tool: {
    "name": "search_engine",
    "description": "Search for information that will aid in determining a response to the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": "search keywords"
            }
        },
        "required": ["keywords"]
    }
}. You should summarize the context concisely into several keywords, ensuring that all the essential information is covered comprehensively. Your summary should be informative and beneficial for predicting in-hospital mortality.
'''

keywords_user = Template('''
Here is the context:
{{context}}

Multiple search keywords are allowed as a list of strings and format that in the keywords field as {"keywords": List(Your search queries)} which can be loaded by python json library. Your summary should be informative and beneficial for in-hospital mortality prediction task.

Here is an example of the format you should output:
{"keywords": ["End-Stage Renal Disease", "Hypertensive kidney damage", "Low Albumin level", "Carbon dioxide binding power", "Low Diastolic blood pressure", "Low Blood chlorine"]}

Please respond with only the information you want to search in JSON format without any additional information:
''')

retcare_system = '''You are a helpful medical expert with extensive medical knowledge. I will provide you with electronic health data of an End-Stage Renal Disease (ESRD) patient and some analysis results from our model, with the analysis results including mortality risk and feature importance weight. Your task is to analyze if the model's analysis results are reasonable using the relevant documents. Your analysis should be based on the relevant documents, and do not include any unsupported conclusions.'''

retcare_user = Template('''
Here are the relevant documents:
{{context}}

Here is the healthcare context, including the patient's basic information, analysis results of AI models and similar patients' information:
{{hcontext}}

Note that the analysis results from the AI model are not all correct. Please analyze by following the steps below based on relevant documents:
1. Which of the relevant documents support the AI model's analysis results? Which do not? Please directly cite sentences or paragraphs from the documents' content in your explanation.
2. Do you think whether the AI model's analysis results are reasonable? The prediction of mortality risk is higher or lower than the actual risk? Please provide your analysis based on the relevant documents, and disjudge the analysis results of the AI model if necessary.
3. Please provide your prediction of mortality risk as a number between 0 and 1.

Please think step-by-step and analyze the results based on relevant documents. Do not include any unsupported conclusions. Generate your output formatted as Dict{"result": Result, "explanation": Str(Your analysis)} without any additional information, where result is a number between 0 and 1 indicating the mortality risk prediction:
''')

ensemble_evaluate_system_esrd = '''You are a helpful medical expert with extensive medical knowledge. I will provide you with electronic health data of an End-Stage Renal Disease (ESRD) patient and some analysis results from several AI models. Every model's analysis results include mortality risk and feature importance weight. Your task is to ensemble analysis results of all models and provide your prediction result of mortality risk using the relevant documents.'''

ensemble_evaluate_system_icu = '''You are a helpful medical expert with extensive medical knowledge. I will provide you with electronic health data of an End-Stage Renal Disease (ESRD) patient and some analysis results from several AI models. Every model's analysis results include mortality risk and feature importance weight. Your task is to ensemble analysis results of all models and provide your prediction result of mortality risk using the relevant documents.'''

ensemble_evaluate_user = Template('''
Here are the relevant documents:
{{context}}

Here is the healthcare context, including the patient's basic information, analysis results of AI models and similar patients' information:
{{hcontext}}

Note that the analysis results from the AI model are not all correct. Please think step-by-step and analyze the results based on relevant documents. Generate your output formatted as Dict{"result": Result, "explanation": Str(Your analysis)} without any additional information, where result is a number between 0 and 1 indicating the mortality risk prediction, and explanation is your analysis following the template below:
## Summary
    Please describe your task, summarize the patient's basic information and health status, and restate the AI model's prediction results along with the feature importance and partial statistic information.
## Documents Analysis
    Please analyze the important features identified by the models, determine whether the identified features are reasonable. If they are reasonable, provide and cite relevant literature, include quotations from the sources, and explain the reasoning. If the features are not reasonable, provide and rank important features in your analysis, cite relevant documents and explain. Ensure that the features you identify are present in the dataset.
## Prediction Analysis
    Please evaluate the AI models' prediction results: too low, too high, or reasonable? If it's not reasonable, please provide your own prediction results, represented as a float number between 0 and 1.
''')

ensemble_select_system_esrd = '''You are a helpful medical expert with extensive medical knowledge. I will provide you with electronic health data of an End-Stage Renal Disease (ESRD) patient and some analysis results from several AI models. Every model's analysis results include mortality risk and feature importance weight. Your task is to ensemble analysis results of all models and select one result of models as your prediction result based on the relevant documents.'''

ensemble_select_system_covid = '''You are a helpful medical expert with extensive medical knowledge. I will provide you with electronic health data of a patient admitted with a diagnosis of COVID-19 or suspected COVID-19 infection and some analysis results from several AI models. Every model's analysis results include mortality risk and feature importance weight. Your task is to ensemble analysis results of all models and select one of them as your final result based on the relevant documents.'''

ensemble_select_system_icu = '''You are a helpful medical expert with extensive medical knowledge. I will provide you with electronic health data of a patient in Intensive Care Unit (ICU) and some analysis results from several AI models. Every model's analysis results include mortality risk and feature importance weight. Your task is to ensemble analysis results of all models and select one result of models as your prediction result based on the relevant documents.'''

ensemble_select_user = Template('''
Here are the relevant documents:
{{context}}

Here is the healthcare context, including the patient's basic information, analysis results of AI models and similar patients' information:
{{hcontext}}

Note that the analysis results from the AI model are not all correct. Please think step-by-step and analyze the results based on relevant documents. Generate your output formatted as Dict{"result": Result, "explanation": Str(Your analysis)} without any additional information, where result is a number between 0 and 1 indicating the mortality risk prediction, and explanation is your analysis following the template below:
## Summary
    Please describe your task, summarize the patient's basic information and health status, and restate the AI model's prediction results along with the feature importance and partial statistic information.
## Documents Analysis
    Please analyze the important features identified by the models, determine whether the identified features are reasonable. If they are reasonable, provide and cite relevant literature, include quotations from the sources, and explain the reasoning. If the features are not reasonable, provide and rank important features in your analysis, cite relevant documents and explain. Ensure that the features you identify are present in the dataset.
## Prediction Analysis
    Please evaluate the AI models' prediction results: too low, too high, or reasonable? Choose one of them as your result, represented as a float number between 0 and 1.
''')