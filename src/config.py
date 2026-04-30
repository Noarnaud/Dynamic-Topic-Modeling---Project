"""
Tous les chemins, hyperparamètres et constantes sont définis ici.
"""

import os

# Répertoire racine du projet 
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
METADATA_FILE = os.path.join(DATA_DIR, "archelect_search.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

# Crée les dossiers de sortie s'ils n'existent pas
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# Les 4 années pour lesquelles on a des données
AVAILABLE_YEARS = [73, 78, 81, 88]

# HYPERPARAMÈTRES 
N_TOPICS = 10                   # Nombre de topics pour LDA et BERTopic (comparaison loyale)
N_TOP_WORDS = 10                # Nombre de mots-clés affichés par topic
N_FEATURES_LDA = 2000           # Taille du vocabulaire pour CountVectorizer (LDA)
MIN_DF = 3                      # Le mot doit apparaître dans au moins N documents
MAX_DF = 0.75                   # Ignorer les mots trop fréquents


# EMBEDDINGS
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_BATCH_SIZE = 64


# SPACY
SPACY_MODEL = "fr_core_news_sm"
SPACY_BATCH_SIZE = 50


# COHERENCE
COHERENCE_METRICS = ["c_npmi", "c_v", "u_mass"]

RANDOM_STATE = 42
