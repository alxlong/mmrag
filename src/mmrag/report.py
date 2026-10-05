from sklearn.metrics import cohen_kappa_score
import pandas as pd

def get_retrieval_report(df_eval_run):
    df_per_question_type = df_eval_run.groupby('question_type')[['precision_at_5', 'rr', 'recall']].mean().astype(float)
    df_overall = df_eval_run[['precision_at_5', 'rr', 'recall']].mean().astype(float)

    record = {
            "avg_precision_at_5_easy_lookup": df_per_question_type.loc["easy_lookup", "precision_at_5"],
            "avg_precision_at_5_compositional": df_per_question_type.loc["compositional", "precision_at_5"],
            "avg_precision_at_5_unanswerable": df_per_question_type.loc["unanswerable", "precision_at_5"],
            "mrr_easy_lookup": df_per_question_type.loc["easy_lookup", "rr"],
            "mrr_compositional": df_per_question_type.loc["compositional", "rr"],
            "mrr_unanswerable": df_per_question_type.loc["unanswerable", "rr"],
            "avg_recall_at_20_easy_lookup": df_per_question_type.loc["easy_lookup", "recall"],
            "avg_recall_at_20_compositional": df_per_question_type.loc["compositional", "recall"],
            "avg_recall_at_20_unanswerable": df_per_question_type.loc["unanswerable", "recall"],
            "avg_precision_at_5_overall": df_overall.loc["precision_at_5"],
            "avg_precision_at_5_overall": df_overall.loc["rr"],
            "avg_precision_at_5_overall": df_overall.loc["recall"]
        }

    return record

def get_agreement_report(df_eval_run, df_hd):
    df_eval_run_w_hd = df_eval_run.merge(df_hd, on='id')
    df_ordinal = df_eval_run_w_hd[(df_eval_run_w_hd['human_verdict'] != 'refusal') & (df_eval_run_w_hd['verdict'] != 'refusal')]
    
    record = {
            "n_total": len(df_eval_run_w_hd),
            "n_ordinal": len(df_ordinal),
            "n_refusal_mismatch": int(((df_eval_run_w_hd.human_verdict == "refusal") != (df_eval_run_w_hd.verdict == "refusal")).sum()),
            "agreement_proportion": float((df_eval_run_w_hd["human_verdict"] == df_eval_run_w_hd["verdict"]).mean()),
            "confusion_matrix": pd.crosstab(df_eval_run_w_hd["human_verdict"],  df_eval_run_w_hd["verdict"]).to_dict(),
            "Cohens_kappa": float(cohen_kappa_score(df_ordinal["human_verdict"], df_ordinal["verdict"], weights="quadratic")),
            "Cohens_kappa_by_type": {
                                t: {"n": len(g), "kappa": float(cohen_kappa_score(g["human_verdict"], g["verdict"], weights="quadratic"))}
                                for t, g in df_ordinal.groupby("question_type")
                             }
    }

    return record
