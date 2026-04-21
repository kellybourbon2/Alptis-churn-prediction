"""File to save the model to S3"""
import joblib
import logging
import s3fs

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from Config import MLFLOW_EXPERIMENT_NAME, S3_BUCKET_ARTIFACT_TRAINING
from src.data_processing.data_load import Config  # reuse same S3 config


def save_model_to_s3(
    model,
    model_name: str,
    bucket: str = S3_BUCKET_ARTIFACT_TRAINING,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> None:
    """Explicit S3 dump using s3fs, independent of MLflow.
    Uses the same S3 credentials and pattern as data_load.py.
    """
    local_path = f"/tmp/{model_name}.pkl"
    joblib.dump(model, local_path)
    logging.info(f"Model serialized locally to {local_path}")

    fs = s3fs.S3FileSystem(
        endpoint_url=f"https://{Config.S3_ENDPOINT}",
        key=Config.S3_ACCESS_KEY,
        secret=Config.S3_SECRET_KEY,
        token=Config.S3_SESSION_TOKEN,
        client_kwargs={"verify": Config.S3_VERIFY_SSL},
    )

    s3_path = f"{bucket}/{experiment_name}/{model_name}.pkl"

    try:
        with fs.open(s3_path, "wb") as f:
            with open(local_path, "rb") as local_f:
                f.write(local_f.read())
        logging.info(f"✅ Model saved to S3: {s3_path}")
    except Exception as e:
        logging.error(f"❌ Failed to save model to S3: {e}")
        raise e