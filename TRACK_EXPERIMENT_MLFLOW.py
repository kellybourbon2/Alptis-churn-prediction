#EXPERIMENT listing here to keep track of which variables were tested in each experiment mlflow

MLFLOW_EXPERIMENT_NAME= "kelly_training_with_30_variables"
COLUMNS_TO_KEEP = [
    "client_cotisation_taux_croissance_n_plus1_n",
    "client_revenu_commune_2021",
    "client_nps_churn_mention_n",
    "client_toujours_engage",
    "nb_jours_forfait_journalier",
    "reste_a_charge_pharmacie",
    "client_nps_note_reco_n",
    "nb_decomptes_hospitalisation",
    "nb_soins_optique",
    "nb_decomptes_dentaire",
    "courrier_est_escompte",
    "interaction_service_suivi_du_contrat",
    "impaye_duree_max_action_jours",
    "interaction_motif_autre",
    "interaction_nb_transferts_gestion",
    "remb_alptis_hospitalisation",
    "courrier_segmentation_interne_VADISTE",
    "nb_decomptes_pharmacie",
    "nb_decomptes_divers",
    "reste_a_charge_divers",
    "interaction_nb_total",
    "remb_alptis_divers",
    "reste_a_charge_soins_medicaux",
    "courrier_nb_affaires_n_mois1",
    "client_departement",
    "remb_alptis_pharmacie",
    "courrier_code_partenaire",
    "frais_reels_pharmacie",
    "client_age_souscripteur",
    "remb_alptis",
    "client_cotisations_annualisees_n_plus1"
]

MLFLOW_EXPERIMENT_NAME= "kelly_training_complete_dataset_5_cv"
COLUMNS_TO_DROP = ["client_code_postal", #car on ajoute le revenu médian commune à la place
                   "client_structure_familiale", "client_nom_banque", "client_nps_date_reponse_n_moins1", "client_nps_date_reponse_n", 'client_nps_date_reponse_n_moins1_jours', ]
