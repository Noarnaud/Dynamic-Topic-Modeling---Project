# %%
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import pyLDAvis
import pyLDAvis.lda_model
pyLDAvis.enable_notebook()
from utils import prepare_and_lemmatize_data

# %%

def vectorize(n_features, data):
    """Vectorisation de nos textes lemmatisé

    Arguments: 
        n_features:
        data: dataframe issu de notre fonction utils qui a une colonne `lemmatized_text`.
    
    Return: 
    """
    tf_vectorizer = CountVectorizer(
        max_features=n_features,
        min_df=5,       # Le mot doit apparaître dans au moins 5 documents
        max_df=0.90     # On ignore les mots qui apparaissent dans plus de 90% des documents
    )
    tf = tf_vectorizer.fit_transform(data['lemmatized_text'])
    return tf, tf_vectorizer

def model_LDA(n_topics,tf):
    """Entraine un modèle LDA avec tf issu de `vectorize`

    Arguments:
        n_topics: nombre de thèmes à détecter
        tf: issu de `vectorize`
    Return:
    """
    lda_model = LatentDirichletAllocation(
    n_components=n_topics, 
    max_iter=10,
    random_state=42)
    lda_model.fit(tf)
    return lda_model


def plot_top_words(model, vectorizer, n_top_words, title, nb_lines=2):
    """Affiche les mots les plus impotnats sous forme de graphique en barre.
    """
    feature_names = vectorizer.get_feature_names_out()
    fig, axes = plt.subplots(nb_lines, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 20})
        ax.tick_params(axis="both", which="major", labelsize=15)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
    
    fig.suptitle(title, fontsize=30)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


def interactive_vis(lda_model, tf, tf_vectorizer, output_file):
    vis_data = pyLDAvis.lda_model.prepare(lda_model, tf, tf_vectorizer)
    pyLDAvis.save_html(vis_data, output_file)
    return pyLDAvis.display(vis_data)

# %% 
if __name__ == "__main__" :
    metadata_globale = pd.read_csv("../data/archelect_search.csv")
# On extrait et on lemmatise pour une année spécifique (ex: 1981 pour commencer)
    df_1981 = prepare_and_lemmatize_data(metadata_globale, year=81)
    print(f"Nombre de professions de foi prêtes : {len(df_1981)}")


    # VECTORISATION 
    n_features = 1000 # Nombre maximum de mots à garder dans le vocabulaire
    tf, tf_vectorizer = vectorize(n_features,df_1981)
    print(f"Taille de la matrice document-terme : {tf.shape}")

    n_topics = 10 # Nombre de thématiques politiques à rechercher
    lda_model = model_LDA(n_topics, tf)
    plot_top_words(lda_model, tf_vectorizer, 10, "Topics LDA - Élections Législatives 1981")

    output_file= "lda_1981.html"
    interactive_vis(lda_model, tf, tf_vectorizer, output_file)
# %%
