import os
import json
import pandas as pd

base_dir = "prompt_test_output"
report_folder = "debug_prompt_test_code2"
report_path = os.path.join(base_dir, report_folder)
all_df = []
col_names = ['VOC2007', 'N24News', 'SUN397', 'ObjectNet', 'Country211', 'Place365', 'ImageNet-1K', 'HatefulMemes', 'ImageNet-A', 'ImageNet-R', 'OK-VQA', 'A-OKVQA', 'DocVQA', 'InfographicsVQA', 'ChartQA', 'Visual7W', 'ScienceQA', 'GQA', 'TextVQA', 'VizWiz', 'VisDial', 'CIRR', 'VisualNews_t2i', 'VisualNews_i2t', 'MSCOCO_t2i', 'MSCOCO_i2t', 'NIGHTS', 'WebQA', 'FashionIQ', 'Wiki-SS-NQ', 'OVEN', 'EDIS', 'MSCOCO', 'RefCOCO', 'RefCOCO-Matching', 'Visual7W-Pointing']

for prompt_type in [x for x in os.listdir(report_path)
                    if os.path.isdir(os.path.join(report_path, x))
                    and '.' not in x]:
    results = {}
    prompt_report_path = os.path.join(report_path, prompt_type)
    for dataset in [x for x in os.listdir(prompt_report_path)
                    if "_score.json" in x]:
        dataset_name = dataset.split("_score.json")[0]
        dataset_report_path = os.path.join(prompt_report_path, dataset)
        with open(dataset_report_path, 'r') as f:
            scores = json.load(f)
        results[dataset_name] = scores['hit@1']
        single_df = pd.DataFrame.from_dict(results, orient='index', columns=[prompt_type]).T
    all_df.append(single_df)

df = pd.concat(all_df, axis=0)
df = df[[col for col in col_names if col in df.columns]]
df = df.sort_index()
df = df * 100
df = df.round(2)
df.to_csv(os.path.join(report_path, "summary_results.csv"))
print(df)
percentage_missing = (1 - df.isnull().sum().sum() / df.size) * 100
print(f"Progress: {percentage_missing:.2f}%")