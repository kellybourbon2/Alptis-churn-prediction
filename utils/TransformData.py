# Useful libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from typing import Union
import requests

def make_binary_numeric(col):
    """Fonction robuste qui transforme les colonnes avec exactement deux valeurs uniques 
        en colonnes binaires 0/1: si la colonne est déjà au format 0/1, rien n'est changé """
    if np.issubdtype(col.dtype, np.number) and set(col.dropna().unique()) <= {0, 1}:
        return col.astype(int)
    elif col.nunique() == 2:
        vals = list(col.dropna().unique())
        return (col == vals[0]).astype(int)
    else:
        return col

# Class : Custom context manager - open files
class FileManager:

    extension: str
    df: pd.DataFrame

    def __init__(self, filename: str, sheetname: str = None):
        self.filename = filename
        self.sheetname = sheetname
        self.extension = os.path.splitext(filename)[1]

    def __enter__(self):
        if self.extension.find("xls") != -1:
            if self.sheetname is None:
                raise ValueError("Tab name is missing")
            else:
                self.df = pd.read_excel(self.filename, sheet_name=self.sheetname)
        elif self.extension.find("csv") != -1:
            self.df = pd.read_csv(self.filename)
        return self.df

    def __exit__(self, exc_type, exc_value, traceback):
        self.df = None


# Function : Group DataFrame columns related to their types
def grouptype(data: pd.DataFrame) -> dict:
    '''
    This function groups the columns of a DataFrame with respect to their type.
    '''
    output = {}
    columns = list(data.columns)
    for elem in columns:
        typ = data[elem].dtypes.name
        if typ not in output.keys():
            output[typ] = []
            output[typ].append(elem)
        else:
            output[typ].append(elem)
    return output


# Function : Summarize categorical variables
def catsum(data: pd.DataFrame, varlist: Union[str, list], vardict: dict= None):
    '''
    A function describing categorical variables.
    This function takes: 
    A DataFrame
    A variable or list of variable to describe
    A dictionnary-like variable which describes the dataset variables. Default is None.
    '''
    if isinstance(varlist, str):
        if vardict is not None:
            label = vardict[varlist]
        else:
            label = ''
        percent_missing_values = data[varlist].isna().sum()/data[varlist].shape[0]
        table_values = data[varlist].value_counts(normalize=True).round(4)*100
        top_k = table_values.reset_index().sort_values('proportion', ascending=False).iloc[:5, :]
        wording_top = ''
        for elem in range(top_k.shape[0]):
            wording_top = wording_top + f'{top_k.iloc[elem, 0]}: {top_k.iloc[elem, 1].round(2)}% \n'
        text_to_display = (
            f'------------------- Analyzing {varlist} : {label} \n'
            f'Proportion of missing values : {percent_missing_values:.1%} \n'
            f'Number of unique values : {data[varlist].nunique()} \n'
            f'Top 5 values : \n'
            f'{wording_top} \n\n'
        )
        print(f'{text_to_display}') 
    elif isinstance(varlist, list):
        for i in varlist:
            if vardict is not None:
                label = vardict[i]
            else:
                label = ''
            percent_missing_values = data[i].isna().sum()/data[i].shape[0]
            table_values = data[i].value_counts(normalize=True).round(4)*100
            top_k = table_values.reset_index().sort_values('proportion', ascending=False).iloc[:5, :]
            wording_top = ''
            for elem in range(top_k.shape[0]):
                wording_top = wording_top + f'{top_k.iloc[elem, 0]}: {top_k.iloc[elem, 1].round(2)}% \n'
            text_to_display = (
                f'------------------- Analyzing {i} : {label} \n'
                f'Proportion of missing values : {percent_missing_values:.1%} \n'
                f'Number of unique values : {data[i].nunique()} \n'
                f'Top 5 values : \n'
                f'{wording_top} \n\n'
            )
            print(f'{text_to_display}') 
    else:
        raise ValueError


# Function : Plotting numerical variables
def plotnum(data: pd.DataFrame, numerical_variables_list: Union[str, list]):
    '''
    A function to display main informations about one or a list of numerical variables in a
    Pandas DataFrame.
    '''
    # Evaluate the correctness of datatype:
    if isinstance(numerical_variables_list, str):
        assert isinstance(data[numerical_variables_list].dtype, np.dtypes.Int64DType) or isinstance(data[numerical_variables_list].dtype, np.dtypes.Float64DType), 'Pas une variable numérique'
        print(f'------  Analyse de la variable {numerical_variables_list} \n')
        print(f'Missing values : {100*data[numerical_variables_list].isnull().sum()/data[numerical_variables_list].shape[0]:.1f}% ')
        print(f'Minimum : {data[numerical_variables_list].min():.1f} ')
        print(f'Moyenne : {data[numerical_variables_list].mean():.1f} ')
        print(f'Médiane : {data[numerical_variables_list].median():.1f} ')
        print(f'Maximum : {data[numerical_variables_list].max():.1f} ')
        print(f'Unique values : {data[numerical_variables_list].nunique():.1f}')
        #
        plt.figure(figsize=(16, 7))
        if data[numerical_variables_list].nunique() < 15 :
            nb_bins = data[numerical_variables_list].nunique()
            sns.histplot(data[numerical_variables_list], bins=nb_bins, stat='proportion', discrete=True)
        elif data[numerical_variables_list].nunique() < 1000:
            nb_bins = 20
            sns.histplot(data[numerical_variables_list], bins=nb_bins, stat='proportion')
        elif data[numerical_variables_list].nunique() < 10000:
            nb_bins = 30
            sns.histplot(data[numerical_variables_list], bins=nb_bins, stat='proportion')
        plt.show()

    if isinstance(numerical_variables_list, list):
        for elem in numerical_variables_list:
            assert isinstance(data[elem].dtype, np.dtypes.Int64DType) or isinstance(data[elem].dtype, np.dtypes.Float64DType), f'la variable {elem} n est pas une variable numérique'
            # nb_bins = int((data[elem].max() - data[elem].min())/5)
            print(f'------  Analyse de la variable {elem} \n')
            print(f'Missing values : {100*data[elem].isnull().sum()/data[elem].shape[0]:.1f}% ')
            print(f'Minimum : {data[elem].min():.1f} ')
            print(f'Moyenne : {data[elem].mean():.1f} ')
            print(f'Médiane : {data[elem].median():.1f} ')
            print(f'Maximum : {data[elem].max():.1f} ')
            print(f'Unique values : {data[elem].nunique():.1f}')
            #
            plt.figure(figsize=(16, 7))
            if data[elem].nunique() < 15 :
                nb_bins = data[elem].nunique()
                sns.histplot(data[elem], stat='proportion', discrete=True, binwidth=1)
            elif data[elem].nunique() < 1000:
                # nb_bins = 20
                binwidth = (data[elem].max()-data[elem].min())/20
                sns.histplot(data[elem], binwidth=binwidth, stat='proportion')
            elif data[elem].nunique() < 10000:
                # nb_bins = 30
                binwidth = (data[elem].max()-data[elem].min())/30
                sns.histplot(data[elem], binwidth=binwidth, stat='proportion')
            plt.show()


# Function : Merging DataFrame
def dataframe_merge(input_df1: pd.DataFrame, input_df2: pd.DataFrame, id_1: 'str', id_2: 'str'):
    df_merged = pd.merge(left=input_df1, right=input_df2, how="inner", left_on=id_1, right_on=id_2)
    df_outter = pd.merge(left=input_df1, right=input_df2, how="outer", left_on=id_1, right_on=id_2, indicator=True)
    df_left = df_outter[df_outter["_merge"]=="left_only"]
    df_right = df_outter[df_outter["_merge"]=="right_only"]
    return df_merged, df_left, df_right


# Function : build and plot stacked bars
def stacked_bar(data: pd.DataFrame, variable_root_name: str, x_axis_variable: str, type_stack: str = 'classic'):
    '''
    This function allow to plot a stacked barplot with two variants : classic (figureswith no transformation)
    and frequency (stacked bar with height = 100%).
    It takes a DataFrame, a scheme for the varibales name to use for the bars, and the variable serving as
    X axis. 
    A last parameter is an option that defines the type of bars : 'classic' of 'frequency'
    '''
    list_columns = [c for c in data.columns if c.startswith(variable_root_name)]
    if len(list_columns) == 0:
        raise ValueError('assurez-vous qu il existe dans votre dataset des colonnes avec la racine spécifiée')
    else:
        if type_stack =='classic':
            bottom = None
            fig, ax = plt.subplots(figsize=(16, 7))
            sns.set(style='whitegrid')
            temp_max = data[list_columns].sum(axis=1).max()
            for col in list_columns:
                ax.bar(data[x_axis_variable], data[col], bottom=bottom, label=col, width=0.8)
                bottom = data[col] if bottom is None else bottom + data[col]
            ax.legend(loc='upper left', ncols=int(len(list_columns)/2), fontsize= 'x-small')
            ax.set_ylabel(f'Sum of {variable_root_name}')

            plt.ylim([0, temp_max * 1.1])
            plt.xticks(rotation=45, ha='right')
            plt.show()

            bottom = None
            fig, ax = plt.subplots(figsize=(16, 7))
            # transform data in percentage
            ratio_data = data.copy()
            ratio_data['total_range'] = ratio_data[list_columns].sum(axis=1)
            sns.set(style='whitegrid')
            temp_max = 0
            for col in list_columns:
                ratio_data[col] = (100*ratio_data[col]/ratio_data['total_range']).round(2)
                p = ax.bar(ratio_data[x_axis_variable], ratio_data[col], bottom=bottom, label=col, width=0.8)
                ax.bar_label(p, label_type='center', fmt='%.1f', fontsize= 10)
                bottom = ratio_data[col] if bottom is None else bottom + ratio_data[col]
                temp_max = 100

            plt.ylim([0 , temp_max * 1.1]) 
            ax.legend(loc='upper left', ncols=int(len(list_columns)/2), fontsize= 'x-small')
            ax.set_ylabel(f'frequency of {variable_root_name}')
            plt.xticks(rotation=45, ha='right')
            plt.show()

        elif type_stack == 'frequency':
            bottom = None
            fig, ax = plt.subplots(figsize=(16, 7))
            # transform data in percentage
            ratio_data = data.copy()
            ratio_data['total_range'] = ratio_data[list_columns].sum(axis=1)
            sns.set(style='whitegrid')
            temp_max = 0
            for col in list_columns:
                ratio_data[col] = (100*ratio_data[col]/ratio_data['total_range']).round(2)
                p = ax.bar(ratio_data[x_axis_variable], ratio_data[col], bottom=bottom, label=col, width= 0.8)
                ax.bar_label(p, label_type='center', fmt='%.1f', fontsize= 10)
                bottom = ratio_data[col] if bottom is None else bottom + ratio_data[col]
                temp_max = 100

            plt.ylim([0 , temp_max * 1.1]) 
            ax.legend(loc='upper left', ncols=int(len(list_columns)/2), fontsize= 'x-small')
            ax.set_ylabel(f'frequency of {variable_root_name}')
            plt.xticks(rotation=45, ha='right')
            plt.show()


# Function : Build a dictionnary with correspondance code INSEE et code commune
def code_insee_from_communes_as_dict(code_postal: str):
    '''
    This function takes a zipcode from France and returns the corresponding INSEE code
    This function requires the package request
    '''
    try:
        api = f'https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/correspondance-code-insee-code-postal/records?select=insee_com&where=postal_code%20%3D%20{code_postal}&limit=20'
        r = requests.get(api)
        if r.status_code == 200:
            decrypt = r.json()
            code_insee = decrypt['results'][0]['insee_com']
            return code_postal, code_insee
        else:
            print(f'la requête concernant le code {code_postal} n a pu aboutir')
            code_postal = ''
            code_insee = ''
            return code_postal, code_insee
    except:
        code_postal = ''
        code_insee = ''
        return code_postal, code_insee

    