"""Aggregate secondary files (consommations, reclamations, impayes, interactions)
 by client_code, this file include some features engineering"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2])) #config is added to root

from Config import RECLA_DELAIS_COURT, RECLA_DELAIS_LONG

def aggregate_reclamations(df_reclamations, ref_date):
    """Aggregate reclamations by client: count, motifs, channels, sensitivity, initiator, delays:
        Args: 
            ref_date: the reference date in order 
              to be able to count the days waited since latest reclamation
    """

    df_agg = df_reclamations.groupby("client_code").agg(recla_nombre=("client_code", "count")).reset_index()

    # Top 5 motifs
    df_reclamations = pd.get_dummies(df_reclamations, columns=["recla_motif"], dtype=int)
    motif_cols = [c for c in df_reclamations.columns if c.startswith("recla_motif_") and c not in ["recla_motif_acpr", "recla_motif_sensibilite"]]
    top5 = df_reclamations[motif_cols].sum().nlargest(5).index.tolist()
    others = set(motif_cols) - set(top5)
    
    df_reclamations["recla_motif_Autre"] = df_reclamations[list(others)].sum(axis=1)
    df_motifs = df_reclamations.groupby("client_code")[top5 + ["recla_motif_Autre"]].sum().reset_index()
    df_agg = df_agg.merge(df_motifs, on="client_code", how="left")

    # Primary & secondary channel (mode)
    df_agg = df_agg.merge(
        df_reclamations.groupby("client_code")["recla_canal_entrant"].agg(lambda x: x.value_counts().idxmax()).rename("recla_canal_entrant_principale"),
        on="client_code", how="left"
    )
    
    def get_second_mode(x):
        vc = x.value_counts()
        return vc.index[1] if len(vc) > 1 else None
    
    df_agg = df_agg.merge(
        df_reclamations.groupby("client_code")["recla_canal_entrant"].agg(get_second_mode).rename("recla_canal_entrant_secondaire"),
        on="client_code", how="left"
    )

    # Sensitive reclamations by type
    sensitivity_types = {
        "recla_sensible_resiliation": ["Menace de résiliation"],
        "recla_sensible_tiers": ["Menace saisie d'un tiers"],
        "recla_sensible_courtier": ["Défaut du courtier"],
        "recla_sensible_autre": ['Sur-réclamation - Maintien de contestation', 'Réclamation médiatisée', 'Médiation', 'Complexe et/ou multi canal', 'Réclamation par un tiers']
    }
    
    for col, values in sensitivity_types.items():
        df_reclamations[col] = df_reclamations["recla_motif_sensibilite"].isin(values).astype(int)
    
    df_sensible = df_reclamations.groupby("client_code")[list(sensitivity_types.keys())].sum().reset_index()
    df_agg = df_agg.merge(df_sensible, on="client_code", how="left")

    # Initiated by client
    df_reclamations["recla_initiateur_client"] = df_reclamations["recla_initiateur"].isin(['Client', 'Adhérent / Participant']).astype(int)
    df_agg = df_agg.merge(df_reclamations.groupby("client_code")["recla_initiateur_client"].sum().reset_index(), on="client_code", how="left")

    # Unclosed complaints
    df_reclamations["recla_non_cloturee"] = ((df_reclamations["recla_reponse_gestion"] == "Non") | (df_reclamations["recla_solution_trouvee"] == "Négative")).astype(int)
    df_agg = df_agg.merge(df_reclamations.groupby("client_code")["recla_non_cloturee"].sum().reset_index(), on="client_code", how="left")

    # Delay categories (feature engineering)
    df_reclamations["recla_date_cloture"] = pd.to_datetime(df_reclamations["recla_date_cloture"])
    df_reclamations["recla_date_reception"] = pd.to_datetime(df_reclamations["recla_date_reception"])
    df_reclamations["recla_delais_jours"] = (df_reclamations["recla_date_cloture"] - df_reclamations["recla_date_reception"]).dt.days

    df_reclamations["recla_delais_court"] = (df_reclamations["recla_delais_jours"] <= RECLA_DELAIS_COURT).astype(int)
    df_reclamations["recla_delais_moyen"] = ((df_reclamations["recla_delais_jours"] > RECLA_DELAIS_COURT) & (df_reclamations["recla_delais_jours"] <= RECLA_DELAIS_LONG)).astype(int)
    df_reclamations["recla_delais_long"] = (df_reclamations["recla_delais_jours"] > RECLA_DELAIS_LONG).astype(int)

    df_delais = df_reclamations.groupby("client_code")[["recla_delais_court", "recla_delais_moyen", "recla_delais_long"]].sum().reset_index()
    df_agg = df_agg.merge(df_delais, on="client_code", how="left")

    # Creation of 'recla_latest_jour': durée depuis la dernière reception et la date de prédiction des résiliations
    df_last = (
        df_reclamations.groupby("client_code")["recla_date_reception"]
        .max()
        .reset_index()
        .rename(columns={"recla_date_reception": "recla_latest_jours"})
    )
    df_last["recla_latest_jours"] = (
        ref_date - df_last["recla_latest_jours"]
    ).dt.days

    df_agg = df_agg.merge(df_last, on="client_code", how="left")
    df_agg = df_agg.drop(columns=["recla_date_reception"], errors="ignore")

    return df_agg


def aggregate_consommations(df_consommations):
    """Aggregate consumption by client, by summing all the amounts for one client
       (sum on reste_à_charge, frais_réel, nb_décompte, ...)"""
    
    df = df_consommations.copy()
    df["annee_mois_paiement"] = df["annee_mois_paiement"].astype("string").str.strip()

    cols_num = df.select_dtypes(include=["float64"]).columns.tolist()

    df_agg = df.groupby("client_code").agg(
        {**{col: "sum" for col in cols_num}, "annee_mois_paiement": lambda x: sorted(x.dropna().unique().tolist())}
    ).reset_index()

    return df_agg

def aggregate_interaction(df_interaction):
    """Aggregate interactions by client: total count, motifs, services, channels, transfers, mail history"""
    
    df = df_interaction.copy()

    # Total interactions
    df_agg = df.groupby("client_code").size().to_frame("interaction_nb_total")

    # Top 7 motifs
    top7 = ["Frais courants", "Dentaire", "Hospitalisation", "Télétransmission", "Médecine douce", "Tiers payant", "Optique"]
    df["motif_simplifie"] = df["interaction_motif"].apply(lambda x: x if x in top7 else "interaction_motif_autre")
    
    df_motif = df.groupby(["client_code", "motif_simplifie"]).size().unstack(fill_value=0)
    df_motif.columns = [f"interaction_motif_{col.lower().replace(' ', '_').replace('é', 'e').replace('è', 'e')}" if col != "interaction_motif_autre" else col for col in df_motif.columns]
    df_agg = df_agg.join(df_motif, how="left")

    # Top 3 services
    top3_services = ["Prestations santé", "Suivi du contrat", "Cotisations & Recouvrements"]
    df["service_simplifie"] = df["interaction_service"].apply(lambda x: x if x in top3_services else "interaction_autres_services")
    
    df_service = df.groupby(["client_code", "service_simplifie"]).size().unstack(fill_value=0)
    df_service.columns = [f"interaction_service_{col.lower().replace(' ', '_').replace('é', 'e')}" if col != "interaction_autres_services" else col for col in df_service.columns]
    df_agg = df_agg.join(df_service, how="left")

    # Telephone channel count
    df_tel = df[df["interaction_canal"] == "Téléphone"].groupby("client_code").size().to_frame("interaction_canal_telephone_nb")
    df_agg = df_agg.join(df_tel, how="left")

    # Transfers
    transfer_cols = ["interaction_est_transferee_service_reclamation", "interaction_est_transferee_service_gestion", 
                     "interaction_est_transferee_service_commercial", "interaction_est_externalisee"]
    df_transfer = df.groupby("client_code")[transfer_cols].sum().rename(columns={
        "interaction_est_transferee_service_reclamation": "interaction_nb_transferts_reclamation",
        "interaction_est_transferee_service_gestion": "interaction_nb_transferts_gestion",
        "interaction_est_transferee_service_commercial": "interaction_nb_transferts_commercial",
        "interaction_est_externalisee": "interaction_nb_transferts_externalises"
    })
    df_agg = df_agg.join(df_transfer, how="left")

    # Email history
    df_mail = (
        df[df["interaction_canal"] == "E-mail"]
        .groupby("client_code")[["interaction_date", "interaction_texte_mail"]]
        .apply(lambda x: dict(zip(x["interaction_date"], x["interaction_texte_mail"])) if not x.empty else {},
            include_groups=False)
        .to_frame("interaction_historique_mail")
    )
    df_agg = df_agg.join(df_mail, how="left")

    # Fill NaN with 0 (except mail history)
    for col in df_agg.columns:
        if col != "interaction_historique_mail":
            df_agg[col] = df_agg[col].fillna(0).astype(int)
    df_agg["interaction_historique_mail"] = df_agg["interaction_historique_mail"].apply(
        lambda x: x if isinstance(x, dict) else {})

    return df_agg.reset_index()


def aggregate_impayes(df_impayes):
    """Aggregate overdue payments: count, total amount, action types, max duration"""
    
    df = df_impayes.copy()

    df["impaye_date_debut_action"] = pd.to_datetime(df["impaye_date_debut_action"])
    df["impaye_date_fin_action"] = pd.to_datetime(df["impaye_date_fin_action"])
    df["impaye_duree_action_jours"] = (df["impaye_date_fin_action"] - df["impaye_date_debut_action"]).dt.days

    df_agg = df.groupby("client_code").agg(
        impaye_nb_actions=("client_code", "count"),
        impaye_montant_total=("impaye_montant_cotisation_impayee", "sum"),
        impaye_duree_max_action_jours=("impaye_duree_action_jours", "max")
    ).reset_index()

    # Action types pivot
    df_types = df.groupby(["client_code", "impaye_type_action"]).size().unstack(fill_value=0)
    mapping_actions = {
        "1er Impayé": "impaye_action_1er_impaye",
        "1ere lettre de relance": "impaye_action_1ere_lettre_relance",
        "Mise en demeure": "impaye_action_mise_en_demeure",
        "2ème Impayé": "impaye_action_2eme_impaye",
        "Suspension de garanties": "impaye_action_suspension_garanties",
        "Annonce du contentieux": "impaye_action_annonce_contentieux",
        "En recouvrement": "impaye_action_en_recouvrement"
    }
    df_types.rename(columns=mapping_actions, inplace=True)

    df_agg = df_agg.join(df_types, how="left")
    return df_agg.fillna(0)
