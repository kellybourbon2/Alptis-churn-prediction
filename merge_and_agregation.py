#Ici se trouvent les fonctions d'agrégation des fichiers secondaires ainsi de merge générale des table
#  (impayés, réclamations, interaction, consommations)
#Pour voir l'explication détaillée des démarches employées, se réferer au notebook de chaque fichier

import pandas as pd
from utils.TransformData import make_binary_numeric

def aggregate_reclamations(df_reclamations):
    """
    Agrège les réclamations par client_code avec création d’indicateurs
    (motifs, canaux, sensibilité, initiateur, délais…) selon la méthode présentée dans
    le notebook "reclamation_aggregation.ipynb"

    Arg: 
    df_reclamations: DataFrame, contient le fichier des réclamations non aggrégé

    Returns: 
    df_agg: DataFrame, fichier des réclamations aggrégé
    """

    # ============================================================
    # ÉTAPE 1 — Nombre total de réclamations par client
    # ============================================================
    df_agg = (
        df_reclamations
        .groupby("client_code")
        .agg(recla_nombre=("client_code", "count"))
        .reset_index()
    )

    # ============================================================
    # ÉTAPE 2 — Colonnes sur motifs principaux & canaux principal/secondaire
    # ============================================================

    #One hot encoding of motif columns, and creation of recla_motif_Autre for non top5 motif
    df_reclamations=pd.get_dummies(df_reclamations, columns=["recla_motif"], dtype=int) 
    colonnes_motif = [c for c in df_reclamations.columns if (c.startswith("recla_motif_")) & (c not in ["recla_motif_acpr","recla_motif_sensibilite"])]
    top5 = df_reclamations[colonnes_motif].sum().sort_values(ascending=False).head(5).index.tolist()
    autres = list(set(colonnes_motif) - set(top5))
    df_reclamations["recla_motif_Autre"] = df_reclamations[autres].sum(axis=1)
    colonnes_motif =  top5 + ["recla_motif_Autre","client_code"]
    df_motifs = df_reclamations.groupby("client_code")[top5 + ["recla_motif_Autre"]].sum().reset_index()

    # Canal principal (mode)
    canal_principal = (
        df_reclamations
        .groupby("client_code")["recla_canal_entrant"]
        .agg(lambda x: x.value_counts().idxmax())
        .rename("recla_canal_entrant_principale")
    )

    # Canal secondaire (2e mode)
    def second_mode(x):
        counts = x.value_counts()
        return counts.index[1] if len(counts) > 1 else None

    canal_secondaire = (
        df_reclamations
        .groupby("client_code")["recla_canal_entrant"]
        .agg(second_mode)
        .rename("recla_canal_entrant_secondaire")
    )

    # Fusion des résultats étape 2
    df_agg = (
        df_agg
        .merge(df_motifs, on="client_code", how="left")
        .merge(canal_principal, on="client_code", how="left")
        .merge(canal_secondaire, on="client_code", how="left")
    )

    # ============================================================
    # ÉTAPE 3 — Réclamations sensibles
    # ============================================================
    motif_resiliation = ["Menace de résiliation"]
    motif_tiers = ["Menace saisie d'un tiers"]
    motif_courtier = ["Défaut du courtier"]
    motif_autre = [
        'Sur-réclamation - Maintien de contestation',
        'Réclamation médiatisée',
        'Médiation',
        'Complexe et/ou multi canal',
        'Réclamation par un tiers'
    ]

    df_reclamations["recla_sensible_resiliation"] = df_reclamations["recla_motif_sensibilite"].isin(motif_resiliation).astype(int)
    df_reclamations["recla_sensible_tiers"] = df_reclamations["recla_motif_sensibilite"].isin(motif_tiers).astype(int)
    df_reclamations["recla_sensible_courtier"] = df_reclamations["recla_motif_sensibilite"].isin(motif_courtier).astype(int)
    df_reclamations["recla_sensible_autre"] = df_reclamations["recla_motif_sensibilite"].isin(motif_autre).astype(int)

    df_sensible_agg = df_reclamations.groupby("client_code").agg(
        recla_sensible_resiliation=("recla_sensible_resiliation", "sum"),
        recla_sensible_tiers=("recla_sensible_tiers", "sum"),
        recla_sensible_courtier=("recla_sensible_courtier", "sum"),
        recla_sensible_autre=("recla_sensible_autre", "sum"),
    ).reset_index()

    df_agg = df_agg.merge(df_sensible_agg, on="client_code", how="left")

    # ============================================================
    # ÉTAPE 4 — Nombre de réclamations initiées par le client / adhérent
    # ============================================================
    df_reclamations["recla_initiateur_client"] = df_reclamations["recla_initiateur"].isin(
        ['Client', 'Adhérent / Participant']
    ).astype(int)

    df_initiateur = (
        df_reclamations
        .groupby("client_code")["recla_initiateur_client"]
        .sum()
        .reset_index()
    )

    df_agg = df_agg.merge(df_initiateur, on="client_code", how="left")

    # ============================================================
    # ÉTAPE 5 — Réclamations non clôturées
    # ============================================================
    df_reclamations["recla_non_cloturee"] = (
        (df_reclamations["recla_reponse_gestion"] == "Non") |
        (df_reclamations["recla_solution_trouvee"] == "Négative")
    ).astype(int)

    df_non_cloture = (
        df_reclamations
        .groupby("client_code")["recla_non_cloturee"]
        .sum()
        .reset_index()
    )

    df_agg = df_agg.merge(df_non_cloture, on="client_code", how="left")

    # ============================================================
    # ÉTAPE 6 — Liste des dates de réception
    # ============================================================
    df_dates = df_reclamations.groupby("client_code").agg(
        recla_date_reception=("recla_date_reception", list)
    ).reset_index()

    df_agg = df_agg.merge(df_dates, on="client_code", how="left")

    # ============================================================
    # ÉTAPE 7 — Délais court / moyen / long
    # ============================================================
    df_reclamations["recla_date_cloture"] = pd.to_datetime(df_reclamations["recla_date_cloture"])
    df_reclamations["recla_date_reception"] = pd.to_datetime(df_reclamations["recla_date_reception"])

    df_reclamations["recla_delais"] = df_reclamations["recla_date_cloture"] - df_reclamations["recla_date_reception"]
    df_reclamations["recla_delais_jours"] = df_reclamations["recla_delais"].dt.days

    df_reclamations["recla_delais_court"] = (df_reclamations["recla_delais_jours"] <= 3).astype(int)
    df_reclamations["recla_delais_moyen"] = (
        (df_reclamations["recla_delais_jours"] > 3) &
        (df_reclamations["recla_delais_jours"] <= 15)
    ).astype(int)
    df_reclamations["recla_delais_long"] = (df_reclamations["recla_delais_jours"] > 15).astype(int)

    df_delais_agg = df_reclamations.groupby("client_code").agg(
        recla_delais_court=("recla_delais_court", "sum"),
        recla_delais_moyen=("recla_delais_moyen", "sum"),
        recla_delais_long=("recla_delais_long", "sum")
    ).reset_index()

    df_agg = df_agg.merge(df_delais_agg, on="client_code", how="left")

    # ============================================================
    # FIN — Retour du dataframe agrégé final
    # ============================================================
    return df_agg



def aggregate_consommations(df_consommations):
    """
    Agrège les consommations par client et construit une structure
    d'informations détaillées par date (annee_mois_paiement).

    Méthode d'agrégation :
    - Agrégation par somme pour toutes les colonnes sauf annee_mois_paiement
    - Agrégation en faisant une liste de date pour la colonne annee_mois_paiement
    - Création de la colonne info_par_annee_mois_paiement
        construction d'un dictionnaire imbriqué : date → soin → montants

    Arg: 
    df_consommations: DataFrame, contient le fichier des consommations non aggrégé

    Returns: 
    df_agg: DataFrame, fichier des consommations aggrégé
    """

    # --- Step 1 : Nettoyer la colonne date ---
    df = df_consommations.copy()
    df["annee_mois_paiement"] = df["annee_mois_paiement"].astype("string").str.strip()

    # --- Step 2 : Colonnes numériques à sommer ---
    cols_num = df.select_dtypes(include=["float64"]).columns.tolist()

    # --- Step 3 : Agrégation principale (sommes + dates uniques) ---
    df_agg = (
        df.groupby("client_code")
          .agg(
              {**{col: "sum" for col in cols_num},
               "annee_mois_paiement": lambda x: sorted(x.dropna().unique().tolist())}
          )
          .reset_index()
    )

    # --- Step 4 : Mapping des colonnes de soins ---
    soins = ["hospitalisation", "soins_medicaux", "pharmacie", "optique", "dentaire", "divers"]
    mapping_soins = {
        soin: {
            "frais_reels": f"frais_reels_{soin}",
            "reste_a_charge": f"reste_a_charge_{soin}",
            "remb_alptis": f"remb_alptis_{soin}",
            "remb_ro": f"remb_ro_{soin}",
        }
        for soin in soins
    }

    # --- Step 5 : Fonction pour construire le dict par date ---
    def build_info_dict(df_client):
        info = {}
        for _, row in df_client.iterrows():
            date = row["annee_mois_paiement"]
            if date not in info:
                info[date] = {}

            # vérifier chaque type de soin
            for soin, cols in mapping_soins.items():
                fr  = row[cols["frais_reels"]]
                rac = row[cols["reste_a_charge"]]
                al  = row[cols["remb_alptis"]]
                ro  = row[cols["remb_ro"]]

                # au moins une valeur non nulle → soin présent
                if (fr != 0) or (rac != 0) or (al != 0) or (ro != 0):
                    info[date][soin] = {
                        "frais_reels": fr,
                        "reste_a_charge": rac,
                        "remb_alptis": al,
                        "remb_ro": ro,
                    }

            # supprimer les dates sans aucun soin
            if info[date] == {}:
                del info[date]

        return info

    # --- Step 6 : Appliquer la construction du dict pour chaque client ---
    df_agg["info_par_annee_mois_paiement"] = (
        df.groupby("client_code").apply(build_info_dict).reset_index(drop=True)
    )

    return df_agg


def aggregate_interaction(df_interaction):
    """
    Prend en input le DataFrame des interactions et l'agrège, selon méthode détaillée dans
    agregation_interactions.ipynb. Renvoie le DataFrame agrégé.

    Cette fonction génère, pour chaque client :
    - interaction_nb_total : nombre total d'interaction
    - variables motifs : sept motifs les plus fréquents + 'interaction_motif_autre'
    - variables services : trois services les plus fréquents + 'interaction_autres_services'
    - interaction_canal_telephone_nb : nombre d'interaction effectuées par téléphone
    - variables transferts : comptage des demandes transférées (4 indicateurs)
    - interaction_historique_mail : dictionnaire {date : contenu_mail} pour les interaction e-mail

    Retourne un dataframe final 'df_interaction_agg' contenant toutes ces
    informations, avec client_code en colonne.
    """

    # 1) NB INTERACTIONS TOTAL
    interaction_nb_total = (
        df_interaction
        .groupby("client_code")
        .size()
        .to_frame("interaction_nb_total")
    )

    # 2) MOTIFS (8 variables)
    top7 = [
        "Frais courants",
        "Dentaire",
        "Hospitalisation",
        "Télétransmission",
        "Médecine douce",
        "Tiers payant",
        "Optique"
    ]

    df = df_interaction.copy()
    df["motif_simplifie"] = df["interaction_motif"].apply(
        lambda x: x if x in top7 else "interaction_motif_autre"
    )

    df_motif = (
        df.groupby(["client_code", "motif_simplifie"])
          .size()
          .unstack(fill_value=0)
    )

    df_motif = df_motif.rename(columns={
        "Frais courants": "interaction_motif_frais_courants",
        "Dentaire": "interaction_motif_dentaire",
        "Hospitalisation": "interaction_motif_hospitalisation",
        "Télétransmission": "interaction_motif_teletransmission",
        "Médecine douce": "interaction_motif_medecine_douce",
        "Tiers payant": "interaction_motif_tiers_payant",
        "Optique": "interaction_motif_optique",
        "interaction_motif_autre": "interaction_motif_autre"
    })

    # 3) SERVICES
    top3_services = [
        "Prestations santé",
        "Suivi du contrat",
        "Cotisations & Recouvrements"
    ]

    df["interaction_service_simplifie"] = df["interaction_service"].apply(
        lambda x: x if x in top3_services else "interaction_autres_services"
    )

    df_service = (
        df.groupby(["client_code", "interaction_service_simplifie"])
          .size()
          .unstack(fill_value=0)
    )

    df_service = df_service.rename(columns={
        "Prestations santé": "interaction_service_prestations_sante",
        "Suivi du contrat": "interaction_service_suivi_du_contrat",
        "Cotisations & Recouvrements": "interaction_service_cotisations_recouvrements",
        "interaction_autres_services": "interaction_autres_services"
    })

    # 4) CANAL TELEPHONE
    df_tel = df[df["interaction_canal"] == "Téléphone"]

    df_canal = (
        df_tel.groupby("client_code")
              .size()
              .to_frame("interaction_canal_telephone_nb")
    )

    df_canal["interaction_canal_telephone_nb"] = df_canal["interaction_canal_telephone_nb"].fillna(0)

    # 5) TRANSFERTS
    dummies_transfert = [
        "interaction_est_transferee_service_reclamation",
        "interaction_est_transferee_service_gestion",
        "interaction_est_transferee_service_commercial",
        "interaction_est_externalisee"
    ]

    df_transfert = (
        df.groupby("client_code")[dummies_transfert]
          .sum()
    )

    df_transfert = df_transfert.rename(columns={
        "interaction_est_transferee_service_reclamation": "interaction_nb_transferts_reclamation",
        "interaction_est_transferee_service_gestion": "interaction_nb_transferts_gestion",
        "interaction_est_transferee_service_commercial": "interaction_nb_transferts_commercial",
        "interaction_est_externalisee": "interaction_nb_transferts_externalises"
    })

    # 6) HISTORIQUE MAILS
    df_mail = df[df["interaction_canal"] == "E-mail"]

    if df_mail.empty:
        df_interaction_historique_mail = pd.DataFrame({"interaction_historique_mail": {}})
    else:
        df_interaction_historique_mail = df_mail.groupby("client_code").apply(
            lambda x: dict(zip(x["interaction_date"], x["interaction_texte_mail"]))
        ).to_frame("interaction_historique_mail")

    # 7) JOIN FINAL
    df_interaction_agg = interaction_nb_total.join(df_motif, how="left")
    df_interaction_agg = df_interaction_agg.join(df_service, how="left")
    df_interaction_agg = df_interaction_agg.join(df_canal, how="left")
    df_interaction_agg = df_interaction_agg.join(df_transfert, how="left")
    df_interaction_agg = df_interaction_agg.join(df_interaction_historique_mail, how="left")

    # 8) Nettoyage
    for col in df_interaction_agg.columns:
        if col != "interaction_historique_mail":
            df_interaction_agg[col] = df_interaction_agg[col].fillna(0)

    df_interaction_agg["interaction_historique_mail"] = df_interaction_agg["interaction_historique_mail"].apply(
        lambda x: {} if isinstance(x, float) else x
    )

    # 9) Mettre client_code comme colonne et non index
    df_interaction_agg = df_interaction_agg.reset_index()

    return df_interaction_agg

                            ###  FONCTION FINALE ###
                            ########################


def aggregate_impayes(df_impayes):
    """
    Agrège les données d'impayés au niveau client :
    - impaye_nb_actions
    - impaye_montant_total
    - compteurs pour chaque type d'action (impaye_action_...)
    - impaye_impaye_duree_max_action_jours
    """

    df = df_impayes.copy()

    # Dates en datetime
    df["impaye_date_debut_action"] = pd.to_datetime(df["impaye_date_debut_action"])
    df["impaye_date_fin_action"] = pd.to_datetime(df["impaye_date_fin_action"])

    # Durée de l'action
    df["impaye_duree_action_jours"] = (
        df["impaye_date_fin_action"] - df["impaye_date_debut_action"]
    ).dt.days

    # 1) Nombre total d'actions
    nb_actions = (
        df.groupby("client_code")
          .size()
          .to_frame("impaye_nb_actions")
    )

    # 2) Montant total
    montant_total = (
        df.groupby("client_code")["impaye_montant_cotisation_impayee"]
          .sum()
          .to_frame("impaye_montant_total")
    )

    # 3) Compteurs des types d’action (pivot)
    df_types = (
        df.groupby(["client_code", "impaye_type_action"])
          .size()
          .unstack(fill_value=0)
    )

    # Mapping des noms propres
    mapping_actions = {
        "1er Impayé": "impaye_action_1er_impaye",
        "1ere lettre de relance": "impaye_action_1ere_lettre_relance",
        "Mise en demeure": "impaye_action_mise_en_demeure",
        "2ème Impayé": "impaye_action_2eme_impaye",
        "Suspension de garanties": "impaye_action_suspension_garanties",
        "Annonce du contentieux": "impaye_action_annonce_contentieux",
        "En recouvrement": "impaye_action_en_recouvrement"
    }

    # Renommage
    df_types = df_types.rename(columns=mapping_actions)

    # Si d'autres types existent → renommer proprement
    df_types = df_types.rename(columns=lambda x: x.lower()
                                                 .replace(" ", "_")
                                                 .replace("é", "e")
                                                 .replace("è", "e")
                                                 .replace("ê", "e")
                                                 .replace("à", "a")
                                                 if x not in mapping_actions else x)

    # 4) Durée maximale
    duree_max = (
        df.groupby("client_code")["impaye_duree_action_jours"]
          .max()
          .to_frame("impaye_duree_max_action_jours")
    )

    # Assemblage final
    df_impayes_agg = (
        nb_actions
        .join(montant_total, how="left")
        .join(df_types, how="left")
        .join(duree_max, how="left")
        .reset_index()
    )

    return df_impayes_agg

def portefeuille_cleaning(df_portefeuille):
    """ 
    Take the portefeuille DataFrame as argument. 
    Clean the portefeuille DataFrame based on the 
    observations and methods detailed in the notebook Data_quality_and_cleaning.ipynb.
    """

    df_portefeuille_cleaned = df_portefeuille.copy()

    # 1 - Creating boolean columns
    df_portefeuille_cleaned["client_male_souscripteur"] = make_binary_numeric(df_portefeuille_cleaned["client_sexe_souscripteur"])
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

    return df_portefeuille_cleaned



def create_clean_aggregated_dataset(df_portefeuille, df_consommations, df_reclamations, df_interactions, df_impayes):
    
    """Functions that take all the table as input and : 
    1. Clean the portfolio file 
    2. agregate the secondary files so they only have a unique "client_code" per line
    3. Left-merge all the secondary files with the portefeuille file
    4. Clean the secondary files (filling of na values on certain columns when it makes sens)
    Args: 
       df_portefeuille: DataFrame of portefeuille
       df_consommations: non agregate DataFrame of consommations
       df_reclamations: non agregate DataFrame of consommations
       df_interactions: non agregate DataFrame of consommations
       df_impayes: non agregate DataFrame of consommations
    Returns: 
       Dataframe of agregate and merged tables.

    """
    #1_Data cleaning in portfolio file
    df_portefeuille= portefeuille_cleaning(df_portefeuille)

    #2_agregation of all secondary files
    df_consommations = aggregate_consommations(df_consommations)
    df_reclamations  = aggregate_reclamations(df_reclamations)
    df_impayes       = aggregate_impayes(df_impayes)
    df_interactions  = aggregate_interaction(df_interactions)


    #3_merge of function on client_code
    df = df_portefeuille.copy()
    for other in [df_reclamations, df_consommations, df_impayes, df_interactions]:
        df = df.merge(other, how="left", on="client_code")

    #4_Data cleaning in secondary files

    #for the columns from reclamations, NaN is 0 since it compting values columns only, except recla_canal and recla_date_reception
    for column in df.columns:
        if column.startswith("recla") & (column not in ["recla_canal_entrant_principale","recla_canal_entrant_secondaire","recla_date_reception"]) :
            df[column]=df[column].fillna(0)

    #for the columns from consommations, same
    column_conso = [c for c in df.columns 
                if c.startswith(("frais", "remb", "nb_decomptes"))]
    df[column_conso]=df[column_conso].fillna(0)

    #for the columns from interactions, same
    column_inter=[c for c in df.columns if c.startswith("interaction") and c not in ["interaction_historique_mail"]]
    df[column_inter]=df[column_inter].fillna(0)

    #for the columns from impayés, same
    column_impaye=[c for c in df.columns if c.startswith("impaye")]
    df[column_impaye]=df[column_impaye].fillna(0)

    return df

def aggregate_courtier(df_portefeuille):
    """  This function merges, aggregates and prepares broker-level data by courtier_code_partenaire
        for future analysis and modelling. 
        Takes the dataframe portefeuille as input and returns
         the aggregated broker relative DataFrame

        One important step consists in transforming the percentage of cancellations
        ("resiliation_pct") into a numerical categorical variable ("resiliation_cat").
        To do this, the function defines four cancellation–intensity bands (bins) and
        assigns each of them a numerical code:

            - 0% to 25%   → category 1 (low cancellation rate)
            - 25% to 50%  → category 2 (medium cancellation rate)
            - 50% to 80%  → category 3 (high cancellation rate)
            - 80% to 100% → category 4 (very high cancellation rate)
        """

    #Compute the internal resiliation rate per broker
    dict_client_courtier= df_portefeuille["courtier_code_partenaire"].value_counts().to_dict()

    dict_resiliation_courtier=df.groupby("courtier_code_partenaire").agg({"target_resiliation_6mois": "sum"}).to_dict()["target_resiliation_6mois"]
    dict_pourc_resiliation_courtier= dict()
    for courtier in dict_client_courtier.keys():
        dict_pourc_resiliation_courtier[courtier]= dict_resiliation_courtier[courtier]/dict_client_courtier[courtier]*100
    df_resiliation = pd.DataFrame(list(dict_pourc_resiliation_courtier.items()), columns=["courtier_code_partenaire", "resiliation_pct"])
    # Merge des données réelles
    df_courtier = df_resiliation.merge(df, on="courtier_code_partenaire", how="left")

    # Définition des bins et valeurs numériques
    bins = [0, 25, 50, 80, 100]
    labels_num = [1, 2, 3, 4]

    df_courtier["courtier_resiliation_cat"] = pd.cut(
        df_courtier["resiliation_pct"],
        bins=bins,
        labels=labels_num,
        include_lowest=True
    ).astype(int)

    # Nombre total de clients par courtier
    df_courtier["courtier_nbre_client"] = df_courtier.groupby("courtier_code_partenaire")["client_code"].transform("count")

    # Colonnes à sommer
    colonnes_sum = [
        "courtier_est_escompte",
        "courtier_nb_affaires_n_moins1",
        "courtier_nb_radiations_n_moins1",
        "courtier_nb_affaires_n_moins2",
        "courtier_nb_radiations_n_moins2"
    ]

    # Colonnes uniques
    colonnes_unique = [
        "courtier_reseau_courtage",
        "courtier_segmentation_interne",
        "courtier_type_commission",
        "courtier_anciennete_annees"
    ]

    # Agrégation principale
    df_agg = df_courtier.groupby("courtier_code_partenaire").agg(
        courtier_resiliation_cat=("courtier_resiliation_cat", "first"),
        courtier_nbre_client=("courtier_nbre_client", "first"),
        courtier_anciennete_annees=("courtier_anciennete_annees", "first"),
        courtier_segmentation_interne=("courtier_segmentation_interne", "first")
    ).reset_index()

    # Agrégation secondaire : somme
    df_agg_sum = df_courtier.groupby("courtier_code_partenaire")[colonnes_sum].sum().reset_index()

    # Agrégation secondaire : valeurs uniques
    df_agg_unique = df_courtier.groupby("courtier_code_partenaire")[colonnes_unique].first().reset_index()

    # Assemblage final
    df_agg = (
        df_agg
        .merge(df_agg_sum, on="courtier_code_partenaire", how="left")
        .merge(df_agg_unique, on="courtier_code_partenaire", how="left")
    )

    return df_agg