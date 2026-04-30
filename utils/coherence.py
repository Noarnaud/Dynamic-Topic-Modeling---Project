# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from utils import prepare_and_lemmatize_data

# %% 

def get_lda_topics(model, vectorizer, n_top_words=10):
    """Extrait les mots-clés de chaque Topic du modèle LDA sous forme de liste."""
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_weights in model.components_:
        # argsort trie par ordre croissant, donc on inverse avec [::-1]
        top_indices = np.argsort(topic_weights)[::-1][:n_top_words]
        top_words = [feature_names[i] for i in top_indices]
        topics.append(top_words)
    return topics

def get_bertopic_topics(topic_model, n_top_words=10):
    """Extrait les mots-clés de chaque Topic de BERTopic (en ignorant les outliers -1)."""
    topics = []
    for topic_id in topic_model.get_topics():
        if topic_id != -1: # On ignore la poubelle des textes inclassables
            # get_topic retourne une liste de tuples (mot, probabilité)
            words = [word for word, score in topic_model.get_topic(topic_id)][:n_top_words]
            topics.append(words)
    return topics


# %% 

def calculate_coherence(topics, tokenized_texts, dictionary, metric='c_npmi'):
    """Calcule le score de cohérence global pour une liste de thèmes."""
    coherence_model = CoherenceModel(
        topics=topics,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence=metric
    )
    return coherence_model.get_coherence()


# %% 
if __name__ == "__main__":
    # PRÉPARATION DES DONNÉES
    metadata_globale = pd.read_csv("../data/archelect_search.csv", low_memory=False)
    df_1981 = prepare_and_lemmatize_data(metadata_globale, year=81)
    
    #  CRÉATION DU DICTIONNAIRE GENSIM (Base de l'arbitrage)
    # C'est la référence commune qui permet de juger les deux modèles équitablement
    vectorizer_arbitre = CountVectorizer()
    analyzer = vectorizer_arbitre.build_analyzer()
    
    # On tokenize le texte 
    tokenized_texts = [analyzer(doc) for doc in df_1981['lemmatized_text']]
    dictionary = Dictionary(tokenized_texts)
    
    print("\n" + "="*50)
    print("MATCH DE COHÉRENCE : LDA vs BERTOPIC (1981)")
    print("="*50)

    # ==========================================
    # ENTRAÎNEMENT LDA
    # ==========================================
    print("\n🤖 Entraînement de LDA (10 topics)...")
    tf_vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.90)
    tf = tf_vectorizer.fit_transform(df_1981['lemmatized_text'])
    
    lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
    lda_model.fit(tf)
    
    # Extraction des 10 meilleurs mots pour chaque Topic LDA
    topics_lda = get_lda_topics(lda_model, tf_vectorizer, n_top_words=10)


    # ==========================================
    # ENTRAÎNEMENT BERTOPIC
    # ==========================================
    print("\n🧠 Entraînement de BERTopic (10 topics)...")
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = embedding_model.encode(df_1981['lemmatized_text'].tolist(), show_progress_bar=False)
    
    bertopic_model = BERTopic(language="french", nr_topics=10)
    bertopic_model.fit_transform(df_1981['lemmatized_text'].tolist(), embeddings)
    
    # Extraction des 10 meilleurs mots pour chaque Topic BERTopic
    topics_bertopic = get_bertopic_topics(bertopic_model, n_top_words=10)


    # ==========================================
    # LE VERDICT (Calcul du score NPMI)
    # ==========================================
    print("\n⚖️ Calcul des scores de cohérence (NPMI)...")
    
    # NPMI est généralement considéré comme la meilleure métrique pour l'interprétabilité humaine
    # Les scores vont typiquement de -1 (très mauvais) à +1 (parfait)
    score_lda = calculate_coherence(topics_lda, tokenized_texts, dictionary, metric='c_npmi')
    score_bertopic = calculate_coherence(topics_bertopic, tokenized_texts, dictionary, metric='c_npmi')

    print("\n" + "*"*30)
    print("🏆 RÉSULTATS FINAUX (c_npmi)")
    print("*"*30)
    print(f"Score LDA      : {score_lda:.4f}")
    print(f"Score BERTopic : {score_bertopic:.4f}")
    
    if score_bertopic > score_lda:
        print("\n✅ Hypothèse validée : BERTopic génère des sujets mathématiquement plus cohérents !")
    else:
        print("\n❌ Surprise : LDA reste la meilleure Baseline sur ce corpus pour le moment.")
# %%
