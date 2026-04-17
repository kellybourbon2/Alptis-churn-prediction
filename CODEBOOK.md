# Features engineering/External data/Textual features created (before encoding)

| Variable | Type | Description |
|----------|------|-------------|
| age_categories | str | Age categories |
| courtier_anciennete_categories | str | Broker seniority categories |
| recla_delais_court | int | Number of claims processed within a short delay for `client_code` |
| recla_delais_moyen | int | Number of claims processed within a medium delay for `client_code` |
| recla_delais_long | int | Number of claims processed within a long delay for `client_code` |
| recla_latest_jours | int | Number of days between `ref_date` and the latest claim |
| dernier_paiement_consommation_mois | int | Number of months between `ref_date` and the last payment |
| client_nps_date_reponse_n_moins1_jours | int | Number of days between `ref_date` and the latest NPS feedback |
| client_date_debut_effet_garantie_mois | int | Number of months between `ref_date` and the start of coverage |
| interaction_motif_... | int | 7 variables (one for each of the 7 most frequent reasons): each counts the number of client interactions for the specified reason |
| frais_reel_... | float | One variable per healthcare category: each contains the total actual expenses over the full year for the client and specified category |
| reste_a_charge_... | float | One variable per healthcare category: each contains the total out-of-pocket expenses over the full year for the client and specified category |
| client_revenu_commune_2021 | int | Median income of the municipality where the client lives |
| derniere_interaction_date_mois | int | Count the number of months between the last interaction and the reference date |
|client_date_debut_effet_garantie_mois | int | Count the number of months between the beginning of the contract and the reference date |
|dernier_paiement_consommation | int | Count the number of months between the last consumption refund and the reference date |
|client_cotisation_taux_croissance_n_plus1_n | int | Count the difference between the cotisation between n+1 and n |
|client_cotisation_taux_croissance_n_n_moins1 | int | Count the difference between the cotisation between n and n-1 |
|client_toujours_engage | bool | Indicates whether or not the client is still engaged with Alptis given the age of its contract (if its contracts has less than 11.5 month --> 1) |
|client_nps_churn_mention_n | bool | Indicates whether or not the client has mentioned churn in nps of year n|
|client_nps_price_mention_n | bool | Indicates whether or not the client has mentioned price in nps of year n|
|client_nps_category_n | str | Indicates the client is a detractor, passive or promotor based on nps score of year n|