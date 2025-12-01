from .utils import save_pipeline, save_json
import pandas as pd
from typing import Dict

def save_preprocess_artifacts(pipeline, processed_df: pd.DataFrame, summary: Dict, out_dir: str):
    pipeline_path = f"{out_dir}/preprocessor.pkl"
    data_path = f"{out_dir}/processed.csv"
    summary_path = f"{out_dir}/preprocess_summary.json"

    save_pipeline(pipeline, pipeline_path)
    processed_df.to_csv(data_path, index=False)
    save_json(summary, summary_path)

    return {"pipeline": pipeline_path, "processed_data": data_path, "summary": summary_path}
