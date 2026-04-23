"""Pipeline file that handles the data preparation, without encoding and without normalisation
1. data processing with no encoding and no normalisation and duplication of columns with missing values
2. definition of dataframes for X_train, y_train, X_test, y_test
3. Save each dataframe on S3 storage, in parquet format (with _noenc_nonorm at the end)
 """
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2])) #so src and Config visible

import Config
from src.data_processing.data_processing import DataProcessor
from src.data_processing.data_load import save_data_processed_parquet_to_s3


#loading of training set, with duplication  of columns with missing values
train_processor = DataProcessor(mode="training")

#processing and fit training set
df_train = train_processor.run(optional_encoding=False, optional_normalisation=False)

#loading of validation and evaluation datasets
test_processor = DataProcessor(mode="validation")
eval_processor = DataProcessor(mode="evaluation")

#pass arguments to avoid data-leakage from training to validation
test_processor.scaler                = train_processor.scaler
test_processor.normalized_columns    = train_processor.normalized_columns

#pass arguments from training to evaluation
eval_processor.scaler                = train_processor.scaler
eval_processor.normalized_columns    = train_processor.normalized_columns
eval_processor.encoded_columns   = train_processor.encoded_columns

#processing of validation and evaluation set
df_test = test_processor.run_transform(optional_encoding=False, optional_normalisation=False)
df_eval = eval_processor.run_transform(optional_encoding=False, optional_normalisation=False)

# Split test/validation
X_train = df_train.drop(columns=[Config.TARGET_COLUMN, Config.KEY_COLUMN])
y_train = pd.DataFrame(df_train[Config.TARGET_COLUMN])
X_test  = df_test.drop(columns=[Config.TARGET_COLUMN, Config.KEY_COLUMN])
X_eval  = df_eval
y_test  = pd.DataFrame(df_test[Config.TARGET_COLUMN])

# Save everything on S3 in parquet format
save_data_processed_parquet_to_s3(X_train, "X_train_noenc_nonorm")
save_data_processed_parquet_to_s3(y_train, "y_train_noenc_nonorm")
save_data_processed_parquet_to_s3(X_test, "X_test_noenc_nonorm")
save_data_processed_parquet_to_s3(y_test, "y_test_noenc_nonorm")
save_data_processed_parquet_to_s3(X_eval, "X_eval_noenc_nonorm")