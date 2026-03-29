"Here the file to integrate external data (median life revenue per commune from Insee)"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2])) #so config and external data visible

import pandas as pd
from Config import REVENU_MEDIAN_FRANCE_2021, NEW_COLUMN_REVENU_INSEE

def add_revenu_insee(df, path_insee_commune, new_column_revenu_name= NEW_COLUMN_REVENU_INSEE, revenu_median_fr= REVENU_MEDIAN_FRANCE_2021):
    """Function that takes the merged DataFrame and add the column of median communal revenue
         from Insee, by merging it on the "client_code_postal"
        Args: 
         df: merged dataset (Alptis)
         path_insee: path of the excel files with Insee data on revenu
         new_column_revenu_name: name of the new column added (on median revenu)
        Returns: 
         df: Datafram with new column from Insee
    """
    df_revenu = pd.read_excel(io=path_insee_commune)

    # STEP 1 : Nettoyer les revenus invalides
    invalid = ["N/A - secret statistique", "N/A - résultat non disponible"]


    df_revenu["Médiane du niveau de vie 2021"] = pd.to_numeric(
        df_revenu["Médiane du niveau de vie 2021"].replace(invalid, pd.NA), errors="coerce"
    )

    # STEP 2 : Exploser les codes postaux multiples (ex: "06130/06520" → 2 lignes)
    df_revenu["Code postal"] = df_revenu["Code postal"].astype(str).str.split("/")
    df_revenu = df_revenu.explode("Code postal")
    df_revenu["Code postal"] = df_revenu["Code postal"].str.strip().str.zfill(5)

    # STEP 3 : Remplacer les revenus manquants par la médiane des communes voisines (même préfixe CP)
    df_revenu["_cp_prefix"] = df_revenu["Code postal"].str[:3]
    prefix_median = (df_revenu.groupby("_cp_prefix")["Médiane du niveau de vie 2021"]
                            .median()
                            .rename("_revenu_voisin"))
    df_revenu = df_revenu.merge(prefix_median, on="_cp_prefix", how="left")
    mask = df_revenu["Médiane du niveau de vie 2021"].isna()
    df_revenu.loc[mask, "Médiane du niveau de vie 2021"] = df_revenu.loc[mask, "_revenu_voisin"]
    df_revenu = df_revenu.drop(columns=["_cp_prefix", "_revenu_voisin"])

    # STEP 4 : Merger sur df
    df["client_code_postal"] = df["client_code_postal"].astype(str).str.zfill(5)
    df = (df
        .merge(df_revenu[["Code postal", "Médiane du niveau de vie 2021"]],
            left_on="client_code_postal", right_on="Code postal", how="left")
        .rename(columns={"Médiane du niveau de vie 2021": new_column_revenu_name})
        .drop(columns=["Code postal"])
    )

    #STEP 5: replace rest of missing values with national median income
    df[new_column_revenu_name]= df[new_column_revenu_name].fillna(revenu_median_fr) #info: only concerns 2.52% of dataset
    
    return df