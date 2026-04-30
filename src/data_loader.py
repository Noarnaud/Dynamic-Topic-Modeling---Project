import os
import pickle
import pandas as pd
import re
import spacy
from tqdm import tqdm

from src.config import (
    DATA_DIR, METADATA_FILE, CACHE_DIR,
    SPACY_MODEL, SPACY_BATCH_SIZE, AVAILABLE_YEARS
)


# Load spacy model

try:
    nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
except OSError:
    os.system(f"python -m spacy download {SPACY_MODEL}")
    nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
nlp.max_length = 3_000_000


def select_data(metadata, year=None):
    """Filtre les métadonnées pour une année donnée (format 2 chiffres, ex : 81 pour 1981).

    Arguments:
        metadata: DataFrame avec toutes les métadonnées
        year: année au format 2 chiffres (ex: 81)

    Returns:
        DataFrame filtré pour l'année sélectionnée
    """
    metadata = metadata.copy()
    metadata['year'] = pd.to_datetime(metadata['date'], errors='coerce').dt.year
    metadata['year'] = metadata['year'] % 100
    return metadata[metadata['year'] == year].copy()


def clean_ocr_text(text):
    """Nettoie le texte issu de l'OCR.

    - Recolle les mots coupés par des tirets en fin de ligne
    - Retire les caractères spéciaux
    - Supprime les espaces multiples
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'[^\w\s\.,;:\'\"!?()-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_file(metadata, base_file_path=None):
    """Charge les fichiers textes correspondants aux métadonnées.

    Arguments:
        metadata: DataFrame filtré (doit contenir 'id', 'year', etc.)
        base_file_path: chemin de base vers les dossiers législatives (sans suffixe année)

    Returns:
        DataFrame avec id, candidat, sexe, parti, annee, text
    """
    if base_file_path is None:
        base_file_path = os.path.join(DATA_DIR, "legislatives")

    data_list = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Chargement des textes"):
        file_id = row['id']
        year = int(row['year'])
        folder_name = f"{base_file_path}_{year:02d}"
        file_path = os.path.join(folder_name, f"{file_id}.txt")

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                text = clean_ocr_text(content)

                if len(text) > 0:
                    nom = str(row['titulaire-nom']) if pd.notna(row['titulaire-nom']) else ""
                    prenom = str(row['titulaire-prenom']) if pd.notna(row['titulaire-prenom']) else ""
                    data_list.append({
                        'id': file_id,
                        'candidat': f"{nom} {prenom}".strip(),
                        'sexe': row['titulaire-sexe'],
                        'parti': row['titulaire-liste'],
                        'annee': year,
                        'text': text
                    })
            except Exception as e:
                print(f"Erreur de lecture {file_id}: {e}")

    return pd.DataFrame(data_list)


def prepare_and_lemmatize_data(metadata_df, year, base_file_path=None):
    """Charge et lemmatise les données pour une année donnée.

    Returns:
        DataFrame avec colonnes : id, candidat, sexe, parti, annee, text, lemmatized_text
    """
    if base_file_path is None:
        base_file_path = os.path.join(DATA_DIR, "legislatives")

    metadata_year = select_data(metadata_df, year)
    df = load_file(metadata_year, base_file_path)

    print(f" Lemmatisation de {len(df)} documents...")
    df['lemmatized_text'] = [
        " ".join([
            token.lemma_
            for token in doc
            if not token.is_stop and token.is_alpha and len(token) > 2
        ])
        for doc in nlp.pipe(df['text'], batch_size=SPACY_BATCH_SIZE)
    ]

    return df


# ============================================================
# Chargement multi-années avec cache
# ============================================================

def load_all_years(years=None, use_cache=True):
    """Charge et lemmatise les données pour toutes les années spécifiées.

    Arguments:
        years: liste d'années (format 2 chiffres). Par défaut : AVAILABLE_YEARS
        use_cache: si True, utilise le cache disque pour éviter de re-lemmatiser

    Returns:
        DataFrame global avec toutes les années concaténées
    """
    if years is None:
        years = AVAILABLE_YEARS

    cache_file = os.path.join(CACHE_DIR, f"df_global_{'_'.join(map(str, years))}.pkl")

    # Vérifier le cache
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # Chargement et lemmatisation
    metadata = pd.read_csv(METADATA_FILE, low_memory=False)

    frames = []
    for annee in years:
        print(f"\n--- Préparation de l'année 19{annee} ---")
        df_annee = prepare_and_lemmatize_data(metadata, year=annee)
        frames.append(df_annee)

    df_global = pd.concat(frames, ignore_index=True)

    # Statistiques du corpus
    print(f" STATISTIQUES DU CORPUS")
    print(f"  Total documents : {len(df_global)}")
    for y in years:
        n = len(df_global[df_global['annee'] == y])
        print(f"  Année 19{y:02d}      : {n} documents")
    print(f"  Longueur moyenne : {df_global['text'].str.len().mean():.0f} caractères")
    print(f"  Longueur médiane : {df_global['text'].str.len().median():.0f} caractères")

    # Sauvegarder dans le cache
    if use_cache:
        with open(cache_file, 'wb') as f:
            pickle.dump(df_global, f)
        print(f"\n Cache sauvegardé : {cache_file}")

    return df_global
