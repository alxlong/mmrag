from mmrag.report import get_retrieval_report, get_agreement_report
import pandas as pd
from datetime import datetime
import json

human_eval_df = pd.read_csv('validation/human_eval_run_20261001.csv')
llm_eval_df = pd.read_json('data/outputs/run_20261003.jsonl', lines=True)

retrieval_report = get_retrieval_report(llm_eval_df)
agreement_report = get_agreement_report(llm_eval_df, human_eval_df)

with open(f'data/outputs/run_{datetime.now():%Y%m%d}_retrieval_report.jsonl', 'a') as f:
        f.write(json.dumps(retrieval_report, ensure_ascii=False) + "\n")

with open(f'validation/run_{datetime.now():%Y%m%d}_agreement_report.jsonl', 'a') as f:
        f.write(json.dumps(agreement_report, ensure_ascii=False) + "\n")
