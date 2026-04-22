"""Aggregate secondary files (consommations, reclamations, impayes, interactions)
by client_code, this file includes some features engineering"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from Config import RECLA_DELAIS_COURT, RECLA_DELAIS_LONG, FIXED_CATEGORIES


def aggregate_reclamations(df_reclamations, ref_date):
    """Aggregate reclamations by client: count, ALL motifs (one column each), 
    ALL canal values (one-hot), sensitivity, initiator, delays.
    """

    df = df_reclamations.copy()
    df_agg = df.groupby("client_code").agg(
        recla_nombre=("client_code", "count")
    ).reset_index()

    # --- All motifs (one column per value, no "Autre" bucket) ---
    motif_dummies = pd.get_dummies(df["recla_motif"], prefix="recla_motif", dtype=int)
    # Drop internal technical columns that are not predictive features
    motif_dummies = motif_dummies.drop(
        columns=[c for c in motif_dummies.columns if c in ("recla_motif_acpr", "recla_motif_sensibilite")],
        errors="ignore"
    )
    df_with_motifs = pd.concat([df[["client_code"]], motif_dummies], axis=1)
    df_motifs = df_with_motifs.groupby("client_code")[motif_dummies.columns.tolist()].sum().reset_index()
    df_agg = df_agg.merge(df_motifs, on="client_code", how="left")

    # --- Primary & secondary canal — one-hot all values, no mode-then-encode trick ---
    # Count occurrences of each canal per client, then derive principal/secondary from counts
    canal_dummies = pd.get_dummies(df["recla_canal_entrant"], prefix="recla_canal_entrant", dtype=int)
    df_canal = pd.concat([df[["client_code"]], canal_dummies], axis=1)
    df_canal_agg = df_canal.groupby("client_code")[canal_dummies.columns.tolist()].sum().reset_index()

    # principale = canal with highest count per client (one-hot flag)
    canal_cols = canal_dummies.columns.tolist()
    df_canal_agg["recla_canal_principale"] = df_canal_agg[canal_cols].idxmax(axis=1)
    df_canal_agg["recla_canal_secondaire"] = df_canal_agg[canal_cols].apply(
        lambda row: sorted(
            [(v, c) for c, v in row.items()], reverse=True
        )[1][1] if (row > 0).sum() > 1 else None,
        axis=1
    )

    # One-hot principale and secondaire with consistent column names
    for suffix, source_col in [("principale", "recla_canal_principale"), ("secondaire", "recla_canal_secondaire")]:
        ohe = pd.get_dummies(df_canal_agg[source_col], prefix=f"recla_canal_entrant_{suffix}", dtype=int)
        df_canal_agg = pd.concat([df_canal_agg, ohe], axis=1)

    drop_cols = ["recla_canal_principale", "recla_canal_secondaire"] + canal_cols
    df_canal_agg = df_canal_agg.drop(columns=drop_cols, errors="ignore")
    df_agg = df_agg.merge(df_canal_agg, on="client_code", how="left")

    # --- Sensitive reclamations by type ---
    sensitivity_types = {
        "recla_sensible_resiliation": ["Menace de résiliation"],
        "recla_sensible_tiers":       ["Menace saisie d'un tiers"],
        "recla_sensible_courtier":    ["Défaut du courtier"],
        "recla_sensible_autre":       [
            "Sur-réclamation - Maintien de contestation", "Réclamation médiatisée",
            "Médiation", "Complexe et/ou multi canal", "Réclamation par un tiers"
        ],
    }
    for col, values in sensitivity_types.items():
        df[col] = df["recla_motif_sensibilite"].isin(values).astype(int)
    df_sensible = df.groupby("client_code")[list(sensitivity_types.keys())].sum().reset_index()
    df_agg = df_agg.merge(df_sensible, on="client_code", how="left")

    # --- Initiated by client ---
    df["recla_initiateur_client"] = df["recla_initiateur"].isin(
        ["Client", "Adhérent / Participant"]
    ).astype(int)
    df_agg = df_agg.merge(
        df.groupby("client_code")["recla_initiateur_client"].sum().reset_index(),
        on="client_code", how="left"
    )

    # --- Unclosed complaints ---
    df["recla_non_cloturee"] = (
        (df["recla_reponse_gestion"] == "Non") |
        (df["recla_solution_trouvee"] == "Négative")
    ).astype(int)
    df_agg = df_agg.merge(
        df.groupby("client_code")["recla_non_cloturee"].sum().reset_index(),
        on="client_code", how="left"
    )

    # --- Delay categories ---
    df["recla_date_cloture"]   = pd.to_datetime(df["recla_date_cloture"])
    df["recla_date_reception"] = pd.to_datetime(df["recla_date_reception"])
    df["recla_delais_jours"]   = (df["recla_date_cloture"] - df["recla_date_reception"]).dt.days

    df["recla_delais_court"] = (df["recla_delais_jours"] <= RECLA_DELAIS_COURT).astype(int)
    df["recla_delais_moyen"] = (
        (df["recla_delais_jours"] > RECLA_DELAIS_COURT) &
        (df["recla_delais_jours"] <= RECLA_DELAIS_LONG)
    ).astype(int)
    df["recla_delais_long"] = (df["recla_delais_jours"] > RECLA_DELAIS_LONG).astype(int)

    df_delais = df.groupby("client_code")[
        ["recla_delais_court", "recla_delais_moyen", "recla_delais_long"]
    ].sum().reset_index()
    df_agg = df_agg.merge(df_delais, on="client_code", how="left")

    # --- Days since latest reclamation ---
    df_last = (
        df.groupby("client_code")["recla_date_reception"]
        .max().reset_index()
        .rename(columns={"recla_date_reception": "recla_latest_jours"})
    )
    df_last["recla_latest_jours"] = (ref_date - df_last["recla_latest_jours"]).dt.days
    df_agg = df_agg.merge(df_last, on="client_code", how="left")

    return df_agg


def aggregate_consommations(df_consommations):
    """Aggregate consumption by client — unchanged."""

    df = df_consommations.copy()
    df["annee_mois_paiement"] = df["annee_mois_paiement"].astype("string").str.strip()
    cols_num = df.select_dtypes(include=["float64"]).columns.tolist()

    df_agg = df.groupby("client_code").agg(
        {**{col: "sum" for col in cols_num},
         "annee_mois_paiement": lambda x: sorted(x.dropna().unique().tolist())}
    ).reset_index()

    return df_agg


def aggregate_interaction(df_interaction):
    """Aggregate interactions by client: total count, for all motifs, all services,
    channels, transfers, mail history."""

    df = df_interaction.copy()

    # --- Total interactions ---
    df_agg = df.groupby("client_code").size().to_frame("interaction_nb_total").reset_index()

    motif_dummies = pd.get_dummies(df["interaction_motif"], prefix="interaction_motif", dtype=int)
    df_motif = pd.concat([df[["client_code"]], motif_dummies], axis=1)
    df_motif_agg = df_motif.groupby("client_code")[motif_dummies.columns.tolist()].sum().reset_index()

    # Enforce fixed vocab from config — same columns every time
    if "interaction_motif" in FIXED_CATEGORIES:
        df_motif_agg = df_motif_agg.reindex(
            columns=["client_code"] + FIXED_CATEGORIES["interaction_motif"],
            fill_value=0
        )

    df_agg = df_agg.merge(df_motif_agg, on="client_code", how="left")


    # --- All services (one column per value) ---
    service_dummies = pd.get_dummies(
        df["interaction_service"]
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("é", "e", regex=False),
        prefix="interaction_service",
        dtype=int
    )
    df_service = pd.concat([df[["client_code"]], service_dummies], axis=1)
    df_service_agg = df_service.groupby("client_code")[service_dummies.columns.tolist()].sum().reset_index()
    df_agg = df_agg.merge(df_service_agg, on="client_code", how="left")

    # --- Telephone channel count ---
    df_tel = (
        df[df["interaction_canal"] == "Téléphone"]
        .groupby("client_code").size()
        .to_frame("interaction_canal_telephone_nb")
        .reset_index()
    )
    df_agg = df_agg.merge(df_tel, on="client_code", how="left")

    # --- Transfers ---
    transfer_cols = [
        "interaction_est_transferee_service_reclamation",
        "interaction_est_transferee_service_gestion",
        "interaction_est_transferee_service_commercial",
        "interaction_est_externalisee",
    ]
    df_transfer = (
        df.groupby("client_code")[transfer_cols]
        .sum()
        .rename(columns={
            "interaction_est_transferee_service_reclamation": "interaction_nb_transferts_reclamation",
            "interaction_est_transferee_service_gestion":     "interaction_nb_transferts_gestion",
            "interaction_est_transferee_service_commercial":  "interaction_nb_transferts_commercial",
            "interaction_est_externalisee":                   "interaction_nb_transferts_externalises",
        })
        .reset_index()
    )
    df_agg = df_agg.merge(df_transfer, on="client_code", how="left")

    # Creation of duration variable on interaction_date: "derniere_interaction_date"
    df_date = df.groupby("client_code")["interaction_date"].max().reset_index()
    df_date.rename(columns={"interaction_date":"derniere_interaction_date"}, inplace=True)
    df_agg = df_agg.merge(df_date, on="client_code", how="left")

    #Creation of features on mail texte
    df_mail_churn= df.groupby("client_code")["interaction_mail_churn_mention"].sum().reset_index()
    df_mail_price= df.groupby("client_code")["interaction_mail_price_mention"].sum().reset_index()
    df_mail_cancel_churn= df.groupby("client_code")["interaction_mail_cancel_churn_mention"].sum().reset_index()

    df_agg= df_agg.merge(df_mail_churn, on="client_code", how="left")
    df_agg= df_agg.merge(df_mail_price, on="client_code", how="left")
    df_agg= df_agg.merge(df_mail_cancel_churn, on="client_code", how="left")

    # --- Fill NaN with zero -  except last interaction_date  ---
    non_mail_cols = [c for c in df_agg.columns if c not in ("client_code", "derniere_interaction_date")]
    df_agg[non_mail_cols] = df_agg[non_mail_cols].fillna(0).astype(int)

    return df_agg


def aggregate_impayes(df_impayes):
    """Aggregate overdue payments — unchanged."""

    df = df_impayes.copy()
    df["impaye_date_debut_action"] = pd.to_datetime(df["impaye_date_debut_action"])
    df["impaye_date_fin_action"]   = pd.to_datetime(df["impaye_date_fin_action"])
    df["impaye_duree_action_jours"] = (
        df["impaye_date_fin_action"] - df["impaye_date_debut_action"]
    ).dt.days

    df_agg = df.groupby("client_code").agg(
        impaye_nb_actions=("client_code", "count"),
        impaye_montant_total=("impaye_montant_cotisation_impayee", "sum"),
        impaye_duree_max_action_jours=("impaye_duree_action_jours", "max"),
    ).reset_index()

    df_types = (
        df.groupby(["client_code", "impaye_type_action"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    df_types.columns.name = None
    mapping_actions = {
        "1er Impayé":              "impaye_action_1er_impaye",
        "1ere lettre de relance":  "impaye_action_1ere_lettre_relance",
        "Mise en demeure":         "impaye_action_mise_en_demeure",
        "2ème Impayé":             "impaye_action_2eme_impaye",
        "Suspension de garanties": "impaye_action_suspension_garanties",
        "Annonce du contentieux":  "impaye_action_annonce_contentieux",
        "En recouvrement":         "impaye_action_en_recouvrement",
    }
    df_types.rename(columns=mapping_actions, inplace=True)

    df_agg = df_agg.merge(df_types, on="client_code", how="left")
    return df_agg.fillna(0)