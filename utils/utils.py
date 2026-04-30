import os
import pandas as pd
import re
import spacy
from tqdm import tqdm

try:
    nlp = spacy.load("fr_core_news_sm")
except:
    os.system("python -m spacy download fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")


nlp.max_length = 3000000 
# Ajout de stopword politique
mots_vides_politique = { }
# {
#     "candidat", "candidate", "élection", "vote", "voter", "voix", 
#     "circonscription", "madame", "monsieur", "électeur", "suffrage", 
#     "député", "législatif", "mars", "juin", "mai" # Les mois d'élections
# }
for word in mots_vides_politique:
    nlp.Defaults.stop_words.add(word)


def select_data(metadata, year=None):
    """Select the metadata for selected year (ajoute une colonne year dans le format  '81' pour 1981)
    
    Arguments:
        metadata: dataframe avec toutes les métadonnées
        year: année que l'on veut selectionner
    """    
    metadata['year'] = pd.to_datetime(metadata['date'], errors='coerce').dt.year 
    metadata['year'] = metadata['year'] % 100 
    return metadata[metadata['year'] == year].copy()


def clean_ocr_text(text):
    """
    Nettoie le texte.
    """
    if not isinstance(text, str): 
        return ""
    text = re.sub(r'-\s+', '', text) # Recolle les mots coupés par des tirets en fin de ligne
    text = re.sub(r'[^\w\s\.,;:\'\"!?()-]', ' ', text) # nlever les caractères spéciaux étranges 
    text = re.sub(r'\s+', ' ', text).strip() # Suppression des espaces et sauts de ligne multiples
    return text


def load_file(metadata, base_file_path='../data/legislatives'):
    """Charge les fichiers textes correspondants aux métadonnées.

    Arguments: 
        metadata:
        base_file_path: 

    Return: 
        dataframe avec l'identifiant, le nom/prénom du candidat, le sexe, le parti, l'annee et le texte clean
    """
    data_list = []  
    
    for index, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Chargement des textes"): 
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
        else:
            pass

    return pd.DataFrame(data_list)


def prepare_and_lemmatize_data(metadata_df, year, base_file_path='../data/legislatives', batch_size=50):
    """
    Return un dataframe avec le texte clean, lemmatisé avec les métadonnées pour une année selctionnée.
    """
    metadata_year = select_data(metadata_df, year)
    df = load_file(metadata_year, base_file_path)
    df['lemmatized_text'] = [
        " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha and len(token) > 2]) 
        for doc in nlp.pipe(df['text'], batch_size=batch_size)
    ]
    return df

