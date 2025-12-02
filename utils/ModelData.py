# Useful libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from typing import Union
import requests


# Function : Transform codified categorical variables before ACM
def acm_preprocess_categorical(data: pd.DataFrame, exclude: Union[list, str] = ''):
    '''
    This function takes a pd.DataFrame as input and output a pd.DataFrame with categorical variables
    processed.
    The processing operation involved transforming categorical variables labelled as numerical 
    values into a more self-explaining format.
    Prior to using this function, the columns data types should be correctly defined.
    arguments :
    can take a list of variables to exclude
    can define the type of transformation to apply to each variable. default is quantile
    '''
    # Identify variables to exclude
    data_copy = data.copy()
    if len(exclude) != 0:
        if isinstance(exclude, str):
            columns = list(data.columns)
            if exclude in columns:
                columns.remove(exclude)
                data_copy = data_copy[columns]
            else:
                raise ValueError
        if isinstance(exclude, list):
            columns = list(data.columns)
            for elem in exclude:
                if elem in columns:
                    columns.remove(elem)
                else:
                    print(f'An issue with the col {elem}')
                    raise ValueError
            data_copy = data_copy[columns]

    # Identify columns to transform
    categorical = data_copy.select_dtypes(include=['object']).columns.to_list()

    # Apply transformation
    columns_edited = []
    for elem in categorical:
        unique_values = list(data_copy[elem].unique())
        test_unique_values = sum([not str(x).isdigit() for x in unique_values])
        if test_unique_values == 0:
            mapping = {}
            prefix = elem[:10]
            for iterator in unique_values:
                mapping[iterator] = prefix + '_' + iterator
            data_copy[elem] = data_copy[elem].map(mapping).fillna(data_copy[elem])
            columns_edited.append(elem)

    # Return the new dataframe
    print(f'Liste des colonnes modifiées : {columns_edited}')
    return data_copy


# Function : Transform numerical variables into categories before ACM
def acm_preprocess_numerical(data: pd.DataFrame, exclude: Union[list, str] = ''):
    '''
    This function takes a pd.DataFrame as input and output a pd.DataFrame with categorical variables
    processed.
    The processing operation involved transforming numerical variables into classes 
    based on some function. Default is quantile.
    Prior to using this function, the columns data types should be correctly defined.
    arguments :
    can take a list of variables to exclude
    can define the type of transformation to apply to each variable. default is quantile
    '''
    # Identify variables to exclude
    data_copy = data.copy()
    if len(exclude) != 0:
        if isinstance(exclude, str):
            columns = list(data.columns)
            if exclude in columns:
                columns.remove(exclude)
                data_copy = data_copy[columns]
            else:
                raise ValueError
        if isinstance(exclude, list):
            columns = list(data.columns)
            for elem in exclude:
                if elem in columns:
                    columns.remove(elem)
                else:
                    raise ValueError
            data_copy = data_copy[columns]

    # Identify columns to transform
    numerical = data_copy.select_dtypes(include=['float', 'int']).columns.to_list()

    # Apply transformation
    columns_edited = []
    for elem in numerical:
        # compute quantiles
        #min = data_copy[elem].min()
        #q1 = data_copy[elem].quantile(q=0.2, axis=1)
        #q2 = data_copy[elem].quantile(q=0.4, axis=1)
        #q3 = data_copy[elem].quantile(q=0.6, axis=1)
        #q4 = data_copy[elem].quantile(q=0.8, axis=1)
        #max = data_copy[elem].max()
        prefix = elem[:10] + '_'
        data_copy[elem] = pd.qcut(data_copy[elem], q=5, labels=False, duplicates='drop')
        data_copy[elem] = prefix + data_copy[elem].astype(str)
        columns_edited.append(elem)

    # Return the new dataframe
    print(f'Liste des colonnes modifiées : {columns_edited}')
    return data_copy