import re
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer


class EmailBERTopicNoLeakagePipeline:
    """
    Pipeline BERTopic sans fuite de données :
    - fit sur train uniquement
    - transform sur valid/test
    - agrégation des topics au niveau client
    - jointure avec les autres features tabulaires
    """

    def __init__(
        self,
        embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        min_cluster_size=30,
        min_samples=10,
        min_df=5,
        random_state=42
    ):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric="cosine",
            random_state=random_state
        )

        self.hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        )

        self.vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            min_df=min_df
        )

        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            calculate_probabilities=True,
            verbose=True
        )

        self.is_fitted = False
        self.seen_topic_ids_ = None

    @staticmethod
    def clean_text(text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"\b\d{4,}\b", " ", text)
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def has_email_content(text: str, min_chars: int = 5) -> bool:
        if pd.isna(text):
            return False
        return len(str(text).strip()) >= min_chars

    def _prepare_email_rows(
        self,
        df: pd.DataFrame,
        client_col: str,
        date_col: str,
        text_col: str
    ) -> pd.DataFrame:
        required_cols = [client_col, date_col, text_col]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes : {missing_cols}")

        work = df.copy()
        work = work.reset_index(drop=False).rename(columns={"index": "row_id"})
        work["has_email"] = work[text_col].apply(self.has_email_content)
        work["text_clean"] = work[text_col].apply(self.clean_text)
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")

        email_rows = work[
            (work["has_email"]) &
            (work["text_clean"].str.len() > 0)
        ].copy()

        return email_rows

    def fit_email_topics(
        self,
        df_train: pd.DataFrame,
        client_col: str = "client_code",
        date_col: str = "interaction_date",
        text_col: str = "objet_contenu",
        batch_size: int = 64
    ) -> pd.DataFrame:
        """
        Fit BERTopic sur les emails du train uniquement.
        Retourne le détail des lignes email train avec topic assigné.
        """
        email_rows = self._prepare_email_rows(
            df=df_train,
            client_col=client_col,
            date_col=date_col,
            text_col=text_col
        )

        if email_rows.empty:
            raise ValueError("Aucune ligne email exploitable dans le train.")

        docs = email_rows["text_clean"].tolist()
        embeddings = self.embedding_model.encode(
            docs,
            show_progress_bar=True,
            batch_size=batch_size
        )

        topics, probs = self.topic_model.fit_transform(docs, embeddings)

        email_rows["topic"] = topics

        if probs is not None:
            if len(probs.shape) == 2:
                email_rows["topic_confidence"] = probs.max(axis=1)
            else:
                email_rows["topic_confidence"] = probs
        else:
            email_rows["topic_confidence"] = np.nan

        self.is_fitted = True
        self.seen_topic_ids_ = sorted(pd.Series(topics).dropna().unique().tolist())

        return email_rows

    def transform_email_topics(
        self,
        df_new: pd.DataFrame,
        client_col: str = "client_code",
        date_col: str = "interaction_date",
        text_col: str = "objet_contenu"
    ) -> pd.DataFrame:
        """
        Transforme valid/test avec le modèle appris sur train.
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné via fit_email_topics() avant transform_email_topics().")

        email_rows = self._prepare_email_rows(
            df=df_new,
            client_col=client_col,
            date_col=date_col,
            text_col=text_col
        )

        if email_rows.empty:
            email_rows["topic"] = pd.Series(dtype="int")
            email_rows["topic_confidence"] = pd.Series(dtype="float")
            return email_rows

        docs = email_rows["text_clean"].tolist()
        embeddings = self.embedding_model.encode(docs, show_progress_bar=False)

        topics, probs = self.topic_model.transform(docs, embeddings)

        email_rows["topic"] = topics

        if probs is not None:
            if len(probs.shape) == 2:
                email_rows["topic_confidence"] = probs.max(axis=1)
            else:
                email_rows["topic_confidence"] = probs
        else:
            email_rows["topic_confidence"] = np.nan

        return email_rows

    def get_topic_info(self):
        if not self.is_fitted:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")
        return self.topic_model.get_topic_info()

    def get_topic_words(self, topic_id):
        if not self.is_fitted:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")
        return self.topic_model.get_topic(topic_id)

    def build_client_topic_features(
        self,
        df_topics: pd.DataFrame,
        all_clients_df: pd.DataFrame,
        client_col: str = "client_code",
        date_col: str = "interaction_date"
    ) -> pd.DataFrame:
        """
        Construit les features topics au niveau client.
        Garantit les mêmes colonnes entre train/valid/test
        en s'alignant sur les topics vus au train.
        """
        base_clients = (
            all_clients_df[[client_col]]
            .drop_duplicates()
            .copy()
            .set_index(client_col)
        )

        if df_topics.empty:
            features = base_clients.copy()
            features["n_emails"] = 0
            features["n_topics_distinct"] = 0
            features["mean_topic_confidence"] = 0.0
            features["email_span_days"] = 0
            features["dominant_topic"] = -99
            features["latest_topic"] = -99

            if self.seen_topic_ids_ is not None:
                for t in self.seen_topic_ids_:
                    features[f"topic_count_{t}"] = 0
                    features[f"topic_prop_{t}"] = 0.0

            return features

        work = df_topics.copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")

        # comptes topic/client
        topic_counts = (
            work.groupby([client_col, "topic"])
            .size()
            .unstack(fill_value=0)
        )

        # aligner les colonnes sur les topics vus au train
        if self.seen_topic_ids_ is not None:
            topic_counts = topic_counts.reindex(columns=self.seen_topic_ids_, fill_value=0)

        topic_counts.columns = [f"topic_count_{c}" for c in topic_counts.columns]

        # proportions
        row_sums = topic_counts.sum(axis=1).replace(0, 1)
        topic_props = topic_counts.div(row_sums, axis=0)
        topic_props.columns = [c.replace("topic_count_", "topic_prop_") for c in topic_props.columns]

        # stats générales
        client_stats = (
            work.groupby(client_col)
            .agg(
                n_emails=("row_id", "count"),
                n_topics_distinct=("topic", "nunique"),
                mean_topic_confidence=("topic_confidence", "mean"),
                first_email_date=(date_col, "min"),
                last_email_date=(date_col, "max")
            )
        )

        client_stats["email_span_days"] = (
            client_stats["last_email_date"] - client_stats["first_email_date"]
        ).dt.days.fillna(0)

        client_stats = client_stats.drop(columns=["first_email_date", "last_email_date"], errors="ignore")

        # topic dominant
        dominant_topic = (
            work.groupby([client_col, "topic"])
            .size()
            .reset_index(name="n")
            .sort_values([client_col, "n"], ascending=[True, False])
            .drop_duplicates(subset=[client_col])
            [[client_col, "topic"]]
            .rename(columns={"topic": "dominant_topic"})
            .set_index(client_col)
        )

        # topic le plus récent
        latest_topic = (
            work.sort_values([client_col, date_col])
            .dropna(subset=[date_col])
            .groupby(client_col)
            .tail(1)
            [[client_col, "topic"]]
            .rename(columns={"topic": "latest_topic"})
            .set_index(client_col)
        )

        features = (
            base_clients
            .join(client_stats, how="left")
            .join(topic_counts, how="left")
            .join(topic_props, how="left")
            .join(dominant_topic, how="left")
            .join(latest_topic, how="left")
        )

        for c in features.columns:
            if c in ["dominant_topic", "latest_topic"]:
                features[c] = features[c].fillna(-99)
            elif c.startswith("topic_prop_"):
                features[c] = features[c].fillna(0.0)
            else:
                features[c] = features[c].fillna(0)

        return features

    @staticmethod
    def build_other_client_features(
        df: pd.DataFrame,
        client_col: str = "client_code",
        date_col: str = "interaction_date",
        text_col: str = "objet_contenu",
        target_col: str = None,
        client_level_agg: dict = None
    ) -> pd.DataFrame:
        """
        Agrège les autres variables non email au niveau client.
        """
        exclude_cols = {client_col, date_col, text_col}
        if target_col is not None:
            exclude_cols.add(target_col)

        other_cols = [c for c in df.columns if c not in exclude_cols]

        if client_level_agg is None:
            client_level_agg = {}
            for c in other_cols:
                if pd.api.types.is_numeric_dtype(df[c]):
                    client_level_agg[c] = "mean"
                else:
                    client_level_agg[c] = "first"

            if target_col is not None:
                client_level_agg[target_col] = "max"
        else:
            client_level_agg = client_level_agg.copy()
            if target_col is not None and target_col not in client_level_agg:
                client_level_agg[target_col] = "max"

        base = df.groupby(client_col).agg(client_level_agg)

        # convertir d'éventuelles dates restantes
        datetime_cols = base.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
        for c in datetime_cols:
            base[f"{c}_year"] = base[c].dt.year.fillna(0)
            base[f"{c}_month"] = base[c].dt.month.fillna(0)
            base[f"{c}_day"] = base[c].dt.day.fillna(0)
        base = base.drop(columns=datetime_cols, errors="ignore")

        return base

    def build_dataset_split(
        self,
        df_split: pd.DataFrame,
        email_mode: str,
        client_col: str = "client_code",
        date_col: str = "interaction_date",
        text_col: str = "objet_contenu",
        target_col: str = None,
        client_level_agg: dict = None
    ) -> pd.DataFrame:
        """
        Construit la table client-level pour un split donné.

        email_mode:
            - "fit"       -> pour le train
            - "transform" -> pour valid/test
        """
        if email_mode not in {"fit", "transform"}:
            raise ValueError("email_mode doit valoir 'fit' ou 'transform'.")

        if email_mode == "fit":
            df_topics = self.fit_email_topics(
                df_train=df_split,
                client_col=client_col,
                date_col=date_col,
                text_col=text_col
            )
        else:
            df_topics = self.transform_email_topics(
                df_new=df_split,
                client_col=client_col,
                date_col=date_col,
                text_col=text_col
            )

        topic_features = self.build_client_topic_features(
            df_topics=df_topics,
            all_clients_df=df_split,
            client_col=client_col,
            date_col=date_col
        )

        other_features = self.build_other_client_features(
            df=df_split,
            client_col=client_col,
            date_col=date_col,
            text_col=text_col,
            target_col=target_col,
            client_level_agg=client_level_agg
        )

        final_df = other_features.join(topic_features, how="left")
        final_df = final_df.reset_index()

        return final_df

    @staticmethod
    def align_feature_columns(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        target_col: str = None,
        client_col: str = "client_code"
    ):
        """
        Aligne les colonnes entre train/valid/test.
        Très utile après get_dummies.
        """
        protected_cols = [client_col]
        if target_col is not None:
            protected_cols.append(target_col)

        x_train = train_df.drop(columns=[c for c in protected_cols if c in train_df.columns], errors="ignore")
        x_train = pd.get_dummies(x_train, drop_first=False)

        outputs = [x_train]

        for df_ in [valid_df, test_df]:
            if df_ is None:
                outputs.append(None)
                continue

            x_ = df_.drop(columns=[c for c in protected_cols if c in df_.columns], errors="ignore")
            x_ = pd.get_dummies(x_, drop_first=False)
            x_ = x_.reindex(columns=x_train.columns, fill_value=0)
            outputs.append(x_)

        y_outputs = []
        for df_ in [train_df, valid_df, test_df]:
            if df_ is None or target_col is None or target_col not in df_.columns:
                y_outputs.append(None)
            else:
                y_outputs.append(df_[target_col])

        return outputs[0], outputs[1], outputs[2], y_outputs[0], y_outputs[1], y_outputs[2]