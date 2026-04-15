# Features engineering/External data (before encoding)

| Variable       | Type    | Description                        | 
|----------------|---------|------------------------------------|
| age_categories    | str   | Catégorie d'âges |           
| courtier_anciennete_categories|  str   | Catégories d'ancienneté courtier |
| recla_delais_court|  int  | Nombre de reclamations traitées dans un délais court pour client_code |
| recla_delais_moyen|  int   | Nombre de reclamations traitées dans un délais moyen pour client_code|
| recla_delais_long|  int   | Nombre de reclamations traitées dans un délais long pour client_code |
| recla_latest_jours|int | Temps en jours entre la ref_date et la dernière réclamation |
| dernier_paiement_consommation_mois|int | Temps en mois entre la ref_date et dernier paiement|
| client_nps_date_reponse_n_moins1_jours|int | Temps en jours entre la ref_date et l'avis nps |
| client_date_debut_effet_garantie_mois|int | Temps en mois entre la ref_date et le début des garantie |
| interaction_motif_...|int | 7 variables (7 pour les 7 motifs les + fréquents): chacune compte le nombre d'interaction réalisée par le client pour le motif précisé |
| frais_reel_...|float | Autant de variables que de catégories de soin:  chaque variable compte le nombre total de frais_reels sur toute l'annee pour le client, et la catégorie spécifiée|
| reste_à_charge_...|float | Autant de variables que de catégories de soin:  chaque variable compte le nombre total de reste_a_charge sur toute l'annee pour le client, et la catégorie spécifiée |
| client_revenu_commune_2021...|int | Il s'agit du revenu médian dans la commune de provenance du client|
| last_interaction_date_mois | int | Count the number of months between the last interaction and the reference date |
|client_date_debut_effet_garantie_mois | int | Count the number of months between the beginning of the contract and the reference date |
|dernier_paiement_consommation | int | Count the number of months between the last consumption refund and the reference date |
|client_cotisation_rate_n_plus1_n | int | Count the difference between the cotisation between n+1 and n |
|client_cotisation_rate_n_n_moins1 | int | Count the difference between the cotisation between n and n-1 |