"""Process and clean the portefeuille (portfolio) file"""

import pandas as pd

def portefeuille_cleaning(df_portefeuille):
    """ 
    Take the portefeuille DataFrame as argument. 
    Clean the portefeuille DataFrame based on the 
    observations and methods detailed in the notebook Data_quality_and_cleaning.ipynb.
    """

    df_portefeuille_cleaned = df_portefeuille.copy()

    # 1 - Creating boolean columns
    df_portefeuille_cleaned["client_male_souscripteur"] = (df_portefeuille_cleaned["client_sexe_souscripteur"]=="M").astype(int)
    df_portefeuille_cleaned["client_demenagement_dans_les_12_mois"] = df_portefeuille_cleaned["client_departement_avant_changement_12_mois"].notna().astype(int)
    df_portefeuille_cleaned["courtier_changement_dans_les_12_mois"] = df_portefeuille_cleaned["courtier_code_avant_changement_12_mois"].notna().astype(int)

    # Drop old columns
    df_portefeuille_cleaned.drop(columns=[
        "client_sexe_souscripteur",
        "client_departement_avant_changement_12_mois",
        "courtier_code_avant_changement_12_mois"
    ], inplace=True)

    # 2 - Replacing "00" in client_departement by NaN
    df_portefeuille_cleaned["client_departement"].replace("00", pd.NA, inplace=True)

    # 3 - Imputing missing values in courtier_type_commission based on courtier_code_apporteur
    mode_par_courtier = (
        df_portefeuille_cleaned.groupby("courtier_code_apporteur")["courtier_type_commission"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
    )

    df_portefeuille_cleaned["courtier_type_commission"] = df_portefeuille_cleaned["courtier_type_commission"].fillna(
        df_portefeuille_cleaned["courtier_code_apporteur"].map(mode_par_courtier)
    )

    #4 - encoding of important variables in portefeuille
    df_portefeuille_cleaned = pd.get_dummies(data=df_portefeuille_cleaned, columns= ["client_structure_familiale","courtier_segmentation_interne", "courtier_type_commission", "client_ro_souscripteur","courtier_reseau_courtage"])

    return df_portefeuille_cleaned