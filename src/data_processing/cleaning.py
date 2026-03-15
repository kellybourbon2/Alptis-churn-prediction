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
    df_portefeuille_cleaned.replace({"client_departement": {"00": pd.NA}}, inplace=True)

    # 3 - Imputing missing values in courtier_type_commission based on courtier_code_apporteur
    mode_par_courtier = (
        df_portefeuille_cleaned.groupby("courtier_code_apporteur")["courtier_type_commission"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
    )

    df_portefeuille_cleaned["courtier_type_commission"] = df_portefeuille_cleaned["courtier_type_commission"].fillna(
        df_portefeuille_cleaned["courtier_code_apporteur"].map(mode_par_courtier)
    )

    #4- Imputing NaN in nps score missing with median score
    nps_cols = ["client_nps_note_reco_n", "client_nps_note_reco_n_moins1"]
    df_portefeuille_cleaned.fillna(
        {col: df_portefeuille_cleaned[col].median() for col in nps_cols}, 
        inplace=True
    )

    #5- Fill cotisation columns with 0 when NaN
    cotisation_cols = [c for c in df_portefeuille_cleaned.columns if c.startswith(("client_cotisation"))]
    df_portefeuille_cleaned.fillna({col: 0 for col in cotisation_cols}, inplace=True)
    return df_portefeuille_cleaned

if __name__== "__main__":
    from data_load import data_loading
    df_portefeuille, _, _, _, _= data_loading("training")
    df_portefeuille_cleaned = portefeuille_cleaning(df_portefeuille)
    print(df_portefeuille_cleaned["client_cotisations_annualisees_n_moins2"])

