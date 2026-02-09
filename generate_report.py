import os
import json
import pandas as pd
from tqdm import tqdm

base_dir = "prompt_test_output"
report_folder = "prompt_visdoc_run"
report_path = os.path.join(base_dir, report_folder)
all_df = []
# col_names = ['VOC2007', 'N24News', 'SUN397', 'ObjectNet', 'Country211', 'Place365', 'ImageNet-1K', 'HatefulMemes', 'ImageNet-A', 'ImageNet-R', 'OK-VQA', 'A-OKVQA', 'DocVQA', 'InfographicsVQA', 'ChartQA', 'Visual7W', 'ScienceQA', 'GQA', 'TextVQA', 'VizWiz', 'VisDial', 'CIRR', 'VisualNews_t2i', 'VisualNews_i2t', 'MSCOCO_t2i', 'MSCOCO_i2t', 'NIGHTS', 'WebQA', 'FashionIQ', 'Wiki-SS-NQ', 'OVEN', 'EDIS', 'MSCOCO', 'RefCOCO', 'RefCOCO-Matching', 'Visual7W-Pointing']

col_names = ['ViDoRe_arxivqa',
            'ViDoRe_docvqa',
            'ViDoRe_infovqa',
            'ViDoRe_tabfquad',
            'ViDoRe_tatdqa',
            'ViDoRe_shiftproject',
            'ViDoRe_syntheticDocQA_artificial_intelligence',
            'ViDoRe_syntheticDocQA_energy',
            'ViDoRe_syntheticDocQA_government_reports',
            'ViDoRe_syntheticDocQA_healthcare_industry', 
            'ViDoRe_esg_reports_human_labeled_v2',
            'ViDoRe_biomedical_lectures_v2_multilingual',
            'ViDoRe_economics_reports_v2_multilingual',
            'ViDoRe_esg_reports_v2_multilingual', 
            'ViDoSeek-page',
            'ViDoSeek-doc',
            'MMLongBench-page',
            'MMLongBench-doc', 
            'VisRAG_ArxivQA',
            'VisRAG_ChartQA',
            'VisRAG_MP-DocVQA',
            'VisRAG_SlideVQA',
            'VisRAG_InfoVQA',
            'VisRAG_PlotQA']

for prompt_type in tqdm([x for x in os.listdir(report_path)
                    if os.path.isdir(os.path.join(report_path, x))
                    and '.' not in x], desc='Prompt type:'):
    results = {}
    prompt_report_path = os.path.join(report_path, prompt_type)
    for dataset in [x for x in os.listdir(prompt_report_path)
                    if "_score.json" in x]:
        dataset_name = dataset.split("_score.json")[0]
        dataset_report_path = os.path.join(prompt_report_path, dataset)
        with open(dataset_report_path, 'r') as f:
            scores = json.load(f)
        metrics = metric = "ndcg_linear@5" if "ndcg_linear@5" in scores else "ndcg@5"
        results[dataset_name] = scores[metrics]
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