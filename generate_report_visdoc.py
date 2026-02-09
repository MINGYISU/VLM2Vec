import os
import json
import pandas as pd
from tqdm import tqdm

base_dir = "prompt_test_output"
report_folder = "normal_visdoc_run"
report_path = os.path.join(base_dir, report_folder)
all_df = []

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

results = {}
for dataset in tqdm([x for x in os.listdir(report_path)
                    if "_score.json" in x], desc='Datasets'):
    dataset_name = dataset.split("_score.json")[0]
    dataset_report_path = os.path.join(report_path, dataset)
    with open(dataset_report_path, 'r') as f:
        scores = json.load(f)
    metrics = metric = "ndcg_linear@5" if "ndcg_linear@5" in scores else "ndcg@5"
    results[dataset_name] = scores[metrics]

df = pd.DataFrame.from_dict(results, orient='index', columns=['scores']).T
df = df * 100
df = df.round(2)
df.to_csv(os.path.join(report_path, "summary_results.csv"))
