"""
Module de visualisation 

- Graphiques de synthèse (heatmaps, distributions)
- Sauvegarde batch de toutes les figures
- Style unifié pour le rapport
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'

from src.config import OUTPUT_DIR


# ============================================================
# Distribution des documents par année
# ============================================================

def plot_corpus_distribution(df, title="Distribution per year"):
    """Barplot du nombre de documents par année.

    Returns:
        Figure matplotlib
    """
    counts = df.groupby('annee').size().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        [f"19{y:02d}" for y in counts.index],
        counts.values
    )

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(val))

    ax.set_xlabel("Year of election")
    ax.set_ylabel("Number of manifestos")
    ax.set_title(title)
    ax.grid(axis='y')
    plt.tight_layout()
    return fig


# ============================================================
# Distribution des partis politiques
# ============================================================

def plot_party_distribution(df, top_n=15, title="Most represented political groups"):
    """Barplot horizontal des partis politiques les plus fréquents.

    Returns:
        Figure matplotlib
    """
    party_counts = df['parti'].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(party_counts.index[::-1], party_counts.values[::-1])

    ax.set_xlabel("Number of candidates")
    ax.set_title(title)
    ax.grid(axis='x')
    plt.tight_layout()
    return fig


# ============================================================
# Heatmap topic × année
# ============================================================

def plot_topic_heatmap(topic_model, df, topics, title="Répartition des topics par année"):
    """Heatmap montrant la fréquence de chaque topic par année.

    Arguments:
        topic_model: modèle BERTopic
        df: DataFrame avec colonne 'annee'
        topics: liste des topic IDs assignés à chaque document

    Returns:
        Figure matplotlib
    """
    df_topics = df.copy()
    df_topics['topic'] = topics

    # Exclure les outliers (-1)
    df_topics = df_topics[df_topics['topic'] != -1]

    # Crosstab normalisé par année (proportion)
    ct = pd.crosstab(df_topics['annee'], df_topics['topic'], normalize='index')

    # Récupérer les labels des topics
    topic_info = topic_model.get_topic_info()
    col_labels = []
    for col in ct.columns:
        row = topic_info[topic_info['Topic'] == col]
        if len(row) > 0:
            name = row.iloc[0].get('Name', f'Topic {col}')
            # Tronquer le nom pour lisibilité
            col_labels.append(name[:40])
        else:
            col_labels.append(f"Topic {col}")

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(ct.values, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels([f"19{y:02d}" for y in ct.index], fontsize=12)

    # Ajout des valeurs dans les cellules
    for i in range(len(ct.index)):
        for j in range(len(ct.columns)):
            val = ct.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    fontsize=8, color='black' if val < 0.15 else 'white')

    ax.set_title(title, fontsize=18, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Proportion')
    plt.tight_layout()
    return fig


# ============================================================
# Longueur des textes par année
# ============================================================

def plot_text_length_distribution(df, title="texts lengths distribution"):
    """Boxplot de la longueur des textes par année.

    Returns:
        Figure matplotlib
    """
    df_plot = df.copy()
    df_plot['text_length'] = df_plot['text'].str.len()

    years = sorted(df_plot['annee'].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    data_by_year = [df_plot[df_plot['annee'] == y]['text_length'].values for y in years]
    bp = ax.boxplot(data_by_year, labels=[f"19{y:02d}" for y in years], patch_artist=True)

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    for patch, color in zip(bp['boxes'], colors[:len(years)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Year")
    ax.set_ylabel("Length (characters)")
    ax.set_title(title)
    ax.grid(axis='y')
    plt.tight_layout()
    return fig


# ============================================================
# Sauvegarde batch
# ============================================================

def save_figure(fig, filename, dpi=150):
    """Sauvegarde une figure matplotlib dans le dossier outputs/.

    Arguments:
        fig: Figure matplotlib
        filename: nom du fichier (avec extension)
        dpi: résolution
    """
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Figure sauvegardée : {path}")
    plt.close(fig)
