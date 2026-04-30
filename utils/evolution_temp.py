# %%
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from utils import prepare_and_lemmatize_data

# %% 
if __name__ == "__main__":
    # ==========================================
    # 1. CHARGEMENT DE TOUTES LES ANNÉES
    # ==========================================
    metadata_globale = pd.read_csv("../data/archelect_search.csv", low_memory=False)
    
    # Remplacez cette liste par les années que vous voulez analyser
    # Par exemple, de 1958 à 1993 (format à 2 chiffres comme dans Archelec)
    annees_a_analyser = [58, 62, 67, 68, 73, 78, 81, 88, 93] 
    
    frames = []
    for annee in annees_a_analyser:
        print(f"\n--- Préparation de l'année 19{annee} ---")
        df_annee = prepare_and_lemmatize_data(metadata_globale, year=annee)
        frames.append(df_annee)
        
    # On fusionne toutes les années en un seul grand DataFrame
    df_global = pd.concat(frames, ignore_index=True)
    print(f"\n✅ Corpus global prêt : {len(df_global)} professions de foi au total.")

    # ==========================================
    # 2. VECTORISATION (EMBEDDINGS GLOBAUX)
    # ==========================================
    print("\n🧠 Création des embeddings pour tout le corpus (ça peut prendre un peu de temps)...")
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings_globaux = embedding_model.encode(df_global['lemmatized_text'].tolist(), show_progress_bar=True)

    # ==========================================
    # 3. ENTRAÎNEMENT DU MODÈLE GLOBAL
    # ==========================================
    print("\n🤖 Entraînement du BERTopic global...")
    # On peut autoriser un peu plus de topics (ex: 15 ou 20) car on a beaucoup plus de données
    topic_model = BERTopic(language="french", verbose=True, nr_topics=15)
    topics, probs = topic_model.fit_transform(df_global['lemmatized_text'].tolist(), embeddings_globaux)

    # ==========================================
    # 4. LA MAGIE DU DYNAMIQUE (TOPICS OVER TIME)
    # ==========================================
    print("\n⏳ Calcul de l'évolution temporelle des thèmes...")
    # On extrait la colonne des années qui servira de "timestamp" pour BERTopic
    timestamps = df_global['annee'].tolist()
    
    # Cette fonction calcule la fréquence et l'importance de chaque thème par année
    topics_over_time = topic_model.topics_over_time(
        docs=df_global['lemmatized_text'].tolist(), 
        timestamps=timestamps
    )

    # ==========================================
    # 5. VISUALISATION DE L'ÉVOLUTION
    # ==========================================
    print("\n📈 Génération du graphique d'évolution...")
    fig_dynamic = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
    
    # On sauvegarde ce magnifique graphique
    fig_dynamic.write_html("bertopic_evolution_temporelle.html")
    
    # On l'affiche dans le notebook
    fig_dynamic.show()
    
    # (Optionnel) Pour voir les thèmes globaux
    print(topic_model.get_topic_info().head(10))