"""
BERTopic
- Création des embeddings (Sentence-Transformers)
- Entraînement BERTopic (statique sur une année)
- Entraînement BERTopic dynamique (topics_over_time sur plusieurs années)
- Cache des embeddings pour éviter les recalculs coûteux
"""
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

from src.config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE,
    N_TOPICS, RANDOM_STATE, CACHE_DIR, OUTPUT_DIR
)


# ============================================================
# Embeddings
# ============================================================

def create_embeddings(data, text_column='lemmatized_text', model_name=None):
    """Vectorise les textes avec un modèle Sentence-Transformers.

    Arguments:
        data: DataFrame avec la colonne de texte
        text_column: nom de la colonne à vectoriser
        model_name: nom du modèle HuggingFace (défaut: config)

    Returns:
        np.ndarray d'embeddings (n_docs, dim)
    """
    if model_name is None:
        model_name = EMBEDDING_MODEL_NAME

    embedding_model = SentenceTransformer(model_name)

    print(f"Calcul des embeddings ({len(data)} documents)")
    embeddings = embedding_model.encode(
        data[text_column].tolist(),
        show_progress_bar=True,
        batch_size=EMBEDDING_BATCH_SIZE
    )
    return embeddings


def save_embeddings(embeddings, name="embeddings"):
    """Sauvegarde les embeddings sur disque (format .npy)."""
    path = os.path.join(CACHE_DIR, f"{name}.npy")
    np.save(path, embeddings)
    return path


def load_embeddings(name="embeddings"):
    """Charge les embeddings depuis le cache disque.

    Returns:
        np.ndarray ou None si le cache n'existe pas
    """
    path = os.path.join(CACHE_DIR, f"{name}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def get_or_create_embeddings(data, name="embeddings", use_cache=True, text_column='lemmatized_text'):
    """Charge les embeddings depuis le cache ou les crée.

    Arguments:
        data: DataFrame
        name: nom du fichier cache
        use_cache: utiliser le cache si disponible
        text_column: colonne de texte à vectoriser

    Returns:
        np.ndarray d'embeddings
    """
    if use_cache:
        cached = load_embeddings(name)
        if cached is not None and len(cached) == len(data):
            return cached

    embeddings = create_embeddings(data, text_column)

    if use_cache:
        save_embeddings(embeddings, name)

    return embeddings


# ============================================================
# BERTopic statique
# ============================================================

def train_bertopic(data, embeddings, text_column='lemmatized_text', n_topics=None):
    """Entraîne un modèle BERTopic statique.

    Arguments:
        data: DataFrame
        embeddings: np.ndarray d'embeddings
        text_column: colonne de texte
        n_topics: nombre cible de topics (défaut: N_TOPICS)

    Returns:
        (topic_model, topics, probs)
    """
    if n_topics is None:
        n_topics = N_TOPICS

    print(f"Entraînement BERTopic ({n_topics} topics)")
    topic_model = BERTopic(
        language="french",
        verbose=True,
        nr_topics=n_topics
    )

    topics, probs = topic_model.fit_transform(
        data[text_column].tolist(),
        embeddings
    )

    return topic_model, topics, probs


# ============================================================
# BERTopic dynamique
# ============================================================

def train_dynamic_bertopic(data, embeddings, text_column='lemmatized_text', n_topics=None):
    """Entraîne BERTopic sur tout le corpus et calcule l'évolution temporelle.

    Arguments:
        data: DataFrame global (toutes les années, avec colonne 'annee')
        embeddings: np.ndarray d'embeddings
        text_column: colonne de texte
        n_topics: nombre cible de topics

    Returns:
        (topic_model, topics, probs, topics_over_time)
    """
    if n_topics is None:
        n_topics = N_TOPICS

    # Entraînement du modèle global
    topic_model = BERTopic(
        language="french",
        verbose=True,
        nr_topics=n_topics
    )

    docs = data[text_column].tolist()
    topics, probs = topic_model.fit_transform(docs, embeddings)

    # Calcul de l'évolution temporelle
    timestamps = data['annee'].tolist()
    topics_over_time = topic_model.topics_over_time(
        docs=docs,
        timestamps=timestamps
    )

    return topic_model, topics, probs, topics_over_time


# ============================================================
# Extraction des topics
# ============================================================

def get_bertopic_topics(topic_model, n_top_words=None):
    """Extrait les mots-clés de chaque topic BERTopic (ignore le topic -1 = outliers).

    Returns:
        Liste de listes de mots (un sous-liste par topic)
    """
    from src.config import N_TOP_WORDS
    if n_top_words is None:
        n_top_words = N_TOP_WORDS

    topics = []
    for topic_id in topic_model.get_topics():
        if topic_id != -1:
            words = [word for word, score in topic_model.get_topic(topic_id)][:n_top_words]
            topics.append(words)
    return topics


# ============================================================
# Sauvegarde des visualisations
# ============================================================

def save_bertopic_visualizations(topic_model, topics_over_time=None, docs=None, embeddings=None, prefix="bertopic"):
    """Sauvegarde les visualisations interactives BERTopic en HTML.

    Arguments:
        topic_model: modèle BERTopic entraîné
        topics_over_time: DataFrame d'évolution temporelle (optionnel)
        docs: liste de documents (pour visualize_documents)
        embeddings: embeddings (pour visualize_documents)
        prefix: préfixe des noms de fichiers
    """
    # Barchart des topics
    try:
        fig = topic_model.visualize_barchart(top_n_topics=min(10, len(topic_model.get_topics()) - 1))
        path = os.path.join(OUTPUT_DIR, f"{prefix}_barchart.html")
        fig.write_html(path)
        print(f"Barchart : {path}")
    except Exception as e:
        print(f"Erreur barchart : {e}")

    # Carte des topics
    try:
        fig = topic_model.visualize_topics()
        path = os.path.join(OUTPUT_DIR, f"{prefix}_intertopic.html")
        fig.write_html(path)
        print(f"Intertopic map : {path}")
    except Exception as e:
        print(f"Erreur intertopic : {e}")

    # Évolution temporelle
    if topics_over_time is not None:
        try:
            fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
            fig.update_traces(mode='lines+markers', marker=dict(size=8))
            path = os.path.join(OUTPUT_DIR, f"{prefix}_evolution.html")
            fig.write_html(path)
            print(f"Évolution temporelle : {path}")
        except Exception as e:
            print(f"Erreur évolution : {e}")

    # Carte des documents
    if docs is not None and embeddings is not None:
        try:
            fig = topic_model.visualize_documents(
                docs=docs,
                embeddings=embeddings,
                hide_annotations=True,
                height=600, width=1000
            )
            path = os.path.join(OUTPUT_DIR, f"{prefix}_documents.html")
            fig.write_html(path)
            print(f"Document map : {path}")
        except Exception as e:
            print(f"Erreur document map : {e}")
