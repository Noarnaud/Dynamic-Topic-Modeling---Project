"""
LDA 
- Vectorisation BoW (CountVectorizer)
- Entraînement LDA (scikit-learn)
- Visualisation des mots-clés par topic (barres horizontales)
- Visualisation interactive (pyLDAvis)
- Recherche du nombre optimal de topics via courbe de cohérence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from src.config import (
    N_FEATURES_LDA, N_TOPICS, N_TOP_WORDS,
    MIN_DF, MAX_DF, RANDOM_STATE, OUTPUT_DIR
)


# ============================================================
# Vectorisation
# ============================================================

def vectorize(data, n_features=None):
    """Vectorise les textes lemmatisés avec CountVectorizer (Bag of Words).

    Arguments:
        data: DataFrame avec colonne 'lemmatized_text'
        n_features: taille max du vocabulaire 

    Returns:
        (tf, tf_vectorizer) — matrice document-terme sparse et le vectorizer
    """
    if n_features is None:
        n_features = N_FEATURES_LDA

    tf_vectorizer = CountVectorizer(
        max_features=n_features,
        min_df=MIN_DF,
        max_df=MAX_DF
    )
    tf = tf_vectorizer.fit_transform(data['lemmatized_text'])
    return tf, tf_vectorizer


# ============================================================
# Entraînement LDA
# ============================================================

def train_lda(tf, n_topics=None):
    """Entraîne un modèle LDA sur la matrice document-terme.

    Arguments:
        tf: matrice document-terme issue de vectorize()
        n_topics: nombre de thèmes à rechercher

    Returns:
        Modèle LDA entraîné
    """
    if n_topics is None:
        n_topics = N_TOPICS

    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        random_state=RANDOM_STATE
    )
    lda_model.fit(tf)
    return lda_model


# ============================================================
# Pipeline complète LDA
# ============================================================

def run_lda_pipeline(df, n_topics=None, n_features=None):
    """Pipeline complète : vectorisation + entraînement LDA.

    Returns:
        (lda_model, tf, tf_vectorizer)
    """
    print(f"Vectorisation (max_features={n_features or N_FEATURES_LDA})")
    tf, tf_vectorizer = vectorize(df, n_features)
    print(f"Entraînement LDA ({n_topics or N_TOPICS} topics)")
    lda_model = train_lda(tf, n_topics)

    return lda_model, tf, tf_vectorizer


# ============================================================
# Extraction des topics
# ============================================================

def get_topics(model, vectorizer, n_top_words=None):
    """Extrait les mots-clés de chaque topic LDA.

    Returns:
        Liste de listes de mots
    """
    if n_top_words is None:
        n_top_words = N_TOP_WORDS

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_weights in model.components_:
        top_indices = np.argsort(topic_weights)[::-1][:n_top_words]
        top_words = [feature_names[i] for i in top_indices]
        topics.append(top_words)
    return topics


def get_topic_df(model, vectorizer, n_top_words=None):
    """Retourne un DataFrame propre des topics et leurs mots-clés.

    Returns:
        DataFrame avec colonnes : topic_id, rank, word, weight
    """
    if n_top_words is None:
        n_top_words = N_TOP_WORDS

    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for topic_idx, topic_weights in enumerate(model.components_):
        top_indices = np.argsort(topic_weights)[::-1][:n_top_words]
        for rank, idx in enumerate(top_indices):
            rows.append({
                'topic_id': topic_idx,
                'rank': rank + 1,
                'word': feature_names[idx],
                'weight': topic_weights[idx]
            })
    return pd.DataFrame(rows)


# ============================================================
# Visualisations LDA
# ============================================================

def plot_top_words(model, vectorizer, n_top_words=None, title="Topics LDA", nb_lines=2):
    """Affiche les mots les plus importants sous forme de barres horizontales.

    Arguments:
        model: modèle LDA entraîné
        vectorizer: CountVectorizer
        n_top_words: nombre de mots par topic
        title: titre du graphique
        nb_lines: nombre de lignes du subplot grid
    """
    if n_top_words is None:
        n_top_words = N_TOP_WORDS

    feature_names = vectorizer.get_feature_names_out()
    n_topics = len(model.components_)
    n_cols = 5
    fig, axes = plt.subplots(nb_lines, n_cols, figsize=(30, 15), sharex=True)
    axes = axes.flatten()

    for topic_idx, topic in enumerate(model.components_):
        if topic_idx >= len(axes):
            break
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 20})
        ax.tick_params(axis="both", which="major", labelsize=15)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

    for idx in range(n_topics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=30)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    return fig


def interactive_vis(lda_model, tf, tf_vectorizer, output_file=None):
    """Génère une visualisation interactive pyLDAvis.

    Arguments:
        lda_model: modèle LDA
        tf: matrice document-terme
        tf_vectorizer: CountVectorizer
        output_file: chemin de sauvegarde HTML (optionnel)

    Returns:
        Objet pyLDAvis (affichable dans un notebook)
    """
    import pyLDAvis
    import pyLDAvis.lda_model

    vis_data = pyLDAvis.lda_model.prepare(lda_model, tf, tf_vectorizer)

    if output_file is not None:
        import os
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        pyLDAvis.save_html(vis_data, output_file)
        print(f"pyLDAvis sauvegardé : {output_file}")

    return vis_data
