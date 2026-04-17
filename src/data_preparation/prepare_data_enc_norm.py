"""Pipeline file that handles the data preparation, with encoding and normalisation
1. data processing with encoding and normalisation
2. definition of dataframes for X_train, y_train, X_test, y_test
3. Save each dataframe on S3 storage, in parquet format, with "_enc_norm" at the end
 """
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2])) #so src and Config visible

import Config
from src.data_processing.data_processing import DataProcessor
from src.data_processing.data_load import save_data_processed_parquet_to_s3


#loading of training set
train_processor = DataProcessor(mode="training")

#processing and fit training set
df_train = train_processor.run(optional_encoding=True, optional_normalisation=True)

#loading of validation set
test_processor = DataProcessor(mode="validation")

#pass arguments to avoid data-leakage from training to validation
test_processor.global_means = train_processor.global_means
test_processor.target_encoding_maps = train_processor.target_encoding_maps
test_processor.fixed_categories_fitted = train_processor.fixed_categories_fitted
test_processor.ordinal_maps          = train_processor.ordinal_maps
test_processor.encoded_columns       = train_processor.encoded_columns
test_processor.scaler                = train_processor.scaler
test_processor.normalized_columns    = train_processor.normalized_columns

#processing of validation set
df_test = test_processor.run_transform(optional_encoding=True, optional_normalisation=True)

# Split test/validation
X_train = df_train.drop(columns=[Config.TARGET_COLUMN, Config.KEY_COLUMN])
y_train = pd.DataFrame(df_train[Config.TARGET_COLUMN])
X_test  = df_test.drop(columns=[Config.TARGET_COLUMN, Config.KEY_COLUMN])
y_test  = pd.DataFrame(df_test[Config.TARGET_COLUMN])

# Save everything on S3 in parquet format
save_data_processed_parquet_to_s3(X_train, "X_train_enc_norm")
save_data_processed_parquet_to_s3(y_train, "y_train_enc_norm")
save_data_processed_parquet_to_s3(X_test, "X_test_enc_norm")
save_data_processed_parquet_to_s3(y_test, "y_test_enc_norm")