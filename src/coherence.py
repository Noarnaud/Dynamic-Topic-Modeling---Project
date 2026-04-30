"""
Évaluation quantitative des modèles de topics.

- Calcul des scores de cohérence (c_npmi, c_v, u_mass) via Gensim
- Comparaison LDA vs BERTopic sur les mêmes métriques
- Génération de tableaux et graphiques comparatifs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer

from src.config import COHERENCE_METRICS, N_TOP_WORDS


# ============================================================
# Préparation des données pour Gensim
# ============================================================

def prepare_gensim_data(df, text_column='lemmatized_text'):
    """Tokenize les textes et crée le dictionnaire Gensim.

    Arguments:
        df: DataFrame avec la colonne de texte lemmatisé
        text_column: nom de la colonne

    Returns:
        (tokenized_texts, dictionary)
    """
    vectorizer = CountVectorizer()
    analyzer = vectorizer.build_analyzer()

    tokenized_texts = [analyzer(doc) for doc in df[text_column]]
    dictionary = Dictionary(tokenized_texts)

    return tokenized_texts, dictionary


# ============================================================
# Calcul de cohérence
# ============================================================

def calculate_coherence(topics, tokenized_texts, dictionary, metric='c_npmi'):
    """Calcule le score de cohérence pour une liste de topics.

    Arguments:
        topics: liste de listes de mots (un sous-liste par topic)
        tokenized_texts: liste de listes de tokens
        dictionary: dictionnaire Gensim
        metric: métrique de cohérence ('c_npmi', 'c_v', 'u_mass')

    Returns:
        Score de cohérence (float)
    """
    coherence_model = CoherenceModel(
        topics=topics,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence=metric
    )
    return coherence_model.get_coherence()


# ============================================================
# Comparaison LDA vs BERTopic
# ============================================================

def compare_models(lda_topics, bert_topics, tokenized_texts, dictionary, metrics=None):
    """Compare LDA et BERTopic sur plusieurs métriques de cohérence.

    Arguments:
        lda_topics: liste de listes de mots (LDA)
        bert_topics: liste de listes de mots (BERTopic)
        tokenized_texts: liste de listes de tokens
        dictionary: dictionnaire Gensim
        metrics: liste de métriques (défaut: COHERENCE_METRICS)

    Returns:
        DataFrame avec colonnes : metric, lda_score, bertopic_score, winner
    """
    if metrics is None:
        metrics = COHERENCE_METRICS

    results = []

    for met in metrics:

        try:
            score_lda = calculate_coherence(lda_topics, tokenized_texts, dictionary, met)
        except Exception as e:
            print(f"Erreur LDA ({met}): {e}")
            score_lda = float('nan')

        try:
            score_bert = calculate_coherence(bert_topics, tokenized_texts, dictionary, met)
        except Exception as e:
            print(f"Erreur BERTopic ({met}): {e}")
            score_bert = float('nan')

        winner = "LDA" if score_lda > score_bert else "BERTopic"
        if met == 'u_mass':
            # u_mass : plus grand = meilleur (valeurs négatives, donc plus proche de 0 est mieux)
            winner = "LDA" if score_lda > score_bert else "BERTopic"

        results.append({
            'metric': met,
            'lda_score': score_lda,
            'bertopic_score': score_bert,
            'winner': winner
        })

        print(f"LDA: {score_lda:.4f} | BERTopic: {score_bert:.4f} → {winner}")

    return pd.DataFrame(results)


# ============================================================
# Visualisation de la comparaison
# ============================================================

def plot_coherence_comparison(results_df, title="Cohérence : LDA vs BERTopic"):
    """Graphique en barres comparant LDA et BERTopic sur chaque métrique.

    Arguments:
        results_df: DataFrame issu de compare_models()
        title: titre du graphique

    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results_df))
    width = 0.35

    bars_lda = ax.bar(x - width / 2, results_df['lda_score'], width, label='LDA', color='#4C72B0')
    bars_bert = ax.bar(x + width / 2, results_df['bertopic_score'], width, label='BERTopic', color='#DD8452')

    ax.set_xlabel('Métrique de cohérence', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['metric'], fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Ajout des valeurs sur les barres
    for bar in bars_lda:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    for bar in bars_bert:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig


# ============================================================
# Recherche du nombre optimal de topics (LDA)
# ============================================================

def coherence_vs_k(df, k_range=None, metric='c_npmi', text_column='lemmatized_text'):
    """Calcule la cohérence pour différents nombres de topics (LDA).

    Arguments:
        df: DataFrame avec texte lemmatisé
        k_range: liste du nombre de topics à tester (défaut: [5, 10, 15, 20])
        metric: métrique de cohérence
        text_column: colonne de texte

    Returns:
        DataFrame avec colonnes : k, coherence_score
    """
    from src.lda_model import vectorize, train_lda, get_topics

    if k_range is None:
        k_range = [5, 10, 15, 20]

    tokenized_texts, dictionary = prepare_gensim_data(df, text_column)
    tf, tf_vectorizer = vectorize(df)

    results = []
    for k in k_range:
        print(f"k={k}...", end=" ")
        lda = train_lda(tf, n_topics=k)
        topics = get_topics(lda, tf_vectorizer)
        score = calculate_coherence(topics, tokenized_texts, dictionary, metric)
        results.append({'k': k, 'coherence_score': score})
        print(f"score={score:.4f}")

    return pd.DataFrame(results)


def plot_coherence_vs_k(results_df, metric='c_npmi', title=None):
    """Courbe de cohérence en fonction du nombre de topics.

    Returns:
        Figure matplotlib
    """
    if title is None:
        title = f"Cohérence ({metric}) vs nombre de topics"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df['k'], results_df['coherence_score'], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel("Nombre de topics (k)", fontsize=14)
    ax.set_ylabel(f"Score de cohérence ({metric})", fontsize=14)
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(results_df['k'])
    plt.tight_layout()
    return fig
