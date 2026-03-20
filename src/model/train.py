"""File where the training of the logistic regression is performed"""

from sklearn.model_selection import train_test_split
from data_processing.data_processing import DataProcessor
from data_processing.data_load import data_loading

 # ------------- Pre-process training data -------------
train_processor = DataProcessor(mode="training")
df_train = train_processor.run(optional_normalisation=False)  #fit training set and save arguments

#----------------- Pre-process test data----------------------
test_processor = DataProcessor(mode="evaluation")

#  To do so: Reuse the same pre-processor fitted on training dataset (so no data-leakage with target-encoding, normalisation...)
test_processor.target_encoding_maps = train_processor.target_encoding_maps
test_processor.ordinal_maps          = train_processor.ordinal_maps
test_processor.encoded_columns       = train_processor.encoded_columns
test_processor.scaler                = train_processor.scaler
test_processor.normalized_columns    = train_processor.normalized_columns

df_test = test_processor.run_transform() #pre-process test data using same encoder/scaler

#DROP TARGET/Key here