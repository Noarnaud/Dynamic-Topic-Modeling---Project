# %%
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from utils import prepare_and_lemmatize_data

# %%
def create_embeddings(data, text_column='lemmatized_text'):
    """Vectorisation sémantique avec un modèle Transformer."""
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = embedding_model.encode(data[text_column].tolist(), show_progress_bar=True)
    return embeddings


def model_BERTopic(data, embeddings, text_column='lemmatized_text', n_topics=10):
    """Entraîne le modèle BERTopic statique."""
    print(f"Entraînement de BERTopic (Objectif : ~{n_topics} topics)...")
    
    topic_model = BERTopic(
        language="french", 
        verbose=True,
        nr_topics=n_topics # On force le même nombre de topics que LDA pour comparer loyalement
    )
    
    # Contrairement à LDA, BERTopic prend le texte ET les embeddings en même temps
    topics, probs = topic_model.fit_transform(data[text_column].tolist(), embeddings)
    return topic_model, topics

# %% 
if __name__ == "__main__":
    metadata_globale = pd.read_csv("../data/archelect_search.csv")
    
    df_1981 = prepare_and_lemmatize_data(metadata_globale, year=81)
    print(f"\nNombre de professions de foi prêtes : {len(df_1981)}")

    embeddings = create_embeddings(df_1981, text_column='lemmatized_text')

    n_topics = 10 
    topic_model, topics = model_BERTopic(df_1981, embeddings, 'lemmatized_text', n_topics)


    df_topic_info = topic_model.get_topic_info()
    print("\n=== Récapitulatif des Topics trouvés par BERTopic ===")
    print(df_topic_info.head(11)) 
    # Attention : BERTopic crée toujours un Topic "-1". Ce sont les "outliers" (les textes inclassables).


    fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
    fig_barchart.write_html("bertopic_barchart_1981.html") # On sauvegarde
    fig_barchart.show() # On affiche dans le notebook

    # 6. VISUALISATION INTERACTIVE (L'équivalent de pyLDAvis)
    # Carte des distances entre les thématiques
    fig_intertopic = topic_model.visualize_topics()
    fig_intertopic.write_html("bertopic_intertopic_1981.html")
    fig_intertopic.show()
# %%
