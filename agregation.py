#Ici se trouve les fonctions d'agrégation des fichiers secondaires
#  (impayés, réclamations, interractions, consommations)
import pandas as pd


def agregation_reclamations(df_reclamations):
    """
    Agrège le fichier "réclamations" par client_code 

    Méthode d'agrégation:
      détaillée dans le notebook "démarche_aggregation.ipynb"

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
    # ÉTAPE 2 — Motifs par thème & canaux principal/secondaire
    # ============================================================
    themes = df_reclamations["recla_theme"].dropna().unique()

    def motifs_par_theme(df_client):
        return {
            theme: df_client.loc[df_client["recla_theme"] == theme, "recla_motif"].unique().tolist()
            for theme in themes
        }

    df_theme_motifs = (
        df_reclamations
        .groupby("client_code")
        .apply(motifs_par_theme)
        .apply(pd.Series)
        .add_prefix("recla_theme_")
    )

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
        .merge(df_theme_motifs, on="client_code", how="left")
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

def agregation_consommations(df_consommations):
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