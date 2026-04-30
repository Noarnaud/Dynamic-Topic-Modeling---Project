import pandas as pd
from sklearn.model_selection import train_test_split

# load the corpus
df = pd.read_csv('data/LeMonde2003_9classes.csv')
# select a type of articles to focus the analysis
df = df[df['category'] == 'ART'].sample(n=100).reset_index(drop=True)
print (f"Number of articles in the corpus : {df.shape}")
df.head()

# %%

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import  LatentDirichletAllocation 
import matplotlib.pyplot as plt
import spacy

#! python -m spacy download fr_core_news_sm

def plot_top_words(model, vectorizer, n_top_words, title, nb_lines=2):
    feature_names = vectorizer.get_feature_names_out()
    fig, axes = plt.subplots(nb_lines, 5, figsize=(30, 30), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

n_features = 1000
n_topics = 10
STOPWORDS = [x.strip() for x in open('data/stop_word_fr.txt').readlines()]
nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
df['lemmatized_text'] = [" ".join([token.lemma_ for token in doc]) for doc in nlp.pipe(df['text'])]

# Encode the text with CountVectorizer
tf_vectorizer = CountVectorizer(max_features=n_features)
tf = tf_vectorizer.fit_transform(df['text'])

# Fit LDA model
lda = LatentDirichletAllocation(n_features, random_state=42)
lda.fit(tf)

# visualize the words in topics
plot_top_words(lda, tf_vectorizer, 10, "LDA Topics")

# %%

tf_vectorizer = CountVectorizer(max_features=n_features, stop_words=STOPWORDS)
tf = tf_vectorizer.fit_transform(df['text'])
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(tf)
plot_top_words(lda, tf_vectorizer, 10, "LDA Topics avec Stopwords")

# %%

# La lemmatisation est déjà faite dans le code fourni :
# df['lemmatized_text'] = [" ".join([token.lemma_ for token in doc]) for doc in nlp.pipe(df['text'])]

tf_vectorizer = CountVectorizer(max_features=n_features, stop_words=STOPWORDS)
tf = tf_vectorizer.fit_transform(df['lemmatized_text'])  # ← on utilise le texte lemmatisé
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(tf)
plot_top_words(lda, tf_vectorizer, 10, "LDA Topics avec Lemmatisation"
# %%

import pyLDAvis
import pyLDAvis.lda_model
pyLDAvis.enable_notebook()

pyLDAvis.lda_model.prepare(lda, tf, tf_vectorizer)

# %%

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(df['text'])


nmf = NMF(n_components=10, random_state=42)
W_nmf = nmf.fit_transform(X_tfidf)


plot_top_words(
    model=nmf, 
    vectorizer=tfidf_vectorizer, 
    n_top_words=20, 
    title="NMF avec TF-IDF (Texte brut, pas de stopwords)"
)

feature_names = tfidf_vectorizer.get_feature_names_out()
n_top_words = 20

for topic_idx, topic in enumerate(nmf.components_):
    top_indices = np.argsort(-topic)[:n_top_words]
    top_words = [feature_names[i] for i in top_indices]
    
    print(f"Topic {topic_idx}:")
    print("  " + ", ".join(top_words))
    print()

topic_label = {}
topic_label[0] = 'Théâtre'

# %%


n_features = 1000
n_topics = 10
STOPWORDS = [x.strip() for x in open('data/stop_word_fr.txt').readlines()]
nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
df['lemmatized_text'] = [" ".join([token.lemma_ for token in doc]) for doc in nlp.pipe(df['text'])]


vectorizer = TfidfVectorizer(
    max_features=n_features,  # On garde seulement les 1000 termes les plus pertinents
    stop_words=STOPWORDS,     # On retire la liste de mots vides français
    max_df=0.95,              # ignore les mots présents dans +95% des articles (trop communs)
    min_df=2                  # ignore les mots présents dans moins de 2 articles (fautes de frappe, mots rares)
)
X = vectorizer.fit_transform(df['lemmatized_text'])


nmf = NMF(n_components=10, random_state=42)
W_nmf = nmf.fit_transform(X)


plot_top_words(
    model=nmf, 
    vectorizer=vectorizer, 
    n_top_words=20, 
    title="NMF avec TF-IDF (Texte brut, pas de stopwords)"
)


# %%

# On divise W par la somme de ses lignes. keepdims=True permet de garder le format colonne.
W_normalized = W_nmf / np.sum(W_nmf, axis=1, keepdims=True)

df['dominant_topic'] = np.argmax(W_normalized, axis=1)
df['topic_weight'] = np.max(W_normalized, axis=1)
df[['text', 'dominant_topic', 'topic_weight']].head()

# %%

import matplotlib.pyplot as plt
import textwrap

for i in range(10):
    if i not in topic_label:
        topic_label[i] = f"Sujet inconnu {i}"


topic_names = [f"{k}-{topic_label[k]}" for k in topic_label.keys()]
    
for row in df.sample(n=5, random_state=2026).itertuples():
    
    doc_id = row.Index
    topic_distribution = W_normalized[doc_id]
    dominant_topic = row.dominant_topic
    dominant_score = row.topic_weight
    dominant_label = topic_label[dominant_topic]
    
    print(f"\nDocument {doc_id}")
    print(f"Dominant topic: {dominant_topic} - {dominant_label} ({dominant_score:.3f})\n")
    text = textwrap.fill(row.text, width=100)
    print(text[:1200])
    print("\n\n")
    
    # Topic distribution
    plt.figure(figsize=(12,4))
    bars = plt.bar(topic_names, topic_distribution)
    bars[dominant_topic].set_color("darkred")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Proportion")
    plt.title(f"Topic mixture for document {doc_id}")
    plt.tight_layout()
    plt.show()

    # %%

    from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

docs = df["text"].tolist()

# Encode the documents with SentenceTransformer
# YOUR CODE HERE
sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
docs = df['text'].tolist()
# On génère les embeddings (vecteurs sémantiques)
embeddings = sentence_model.encode(docs, show_progress_bar=True)
# Create a CountVectorizer with French stop words
# YOUR CODE HERE
vectorizer_model = CountVectorizer(stop_words=STOPWORDS)

# Create the BERTopic model using the vectorizer and fit it
# YOUR CODE HERE
topic_model = BERTopic(
    vectorizer_model=vectorizer_model,
    language="multilingual"
)
# On entraîne le modèle sur nos documents et nos embeddings
print("Entraînement de BERTopic en cours...")
topics, probs = topic_model.fit_transform(docs, embeddings)
topic_info = topic_model.get_topic_info()

print("\n--- Informations sur les topics découverts par BERTopic ---")
# On affiche les 15 premiers topics
display(topic_info.head(15))

#%%

topic_model.get_topic_info()
topic_model.visualize_documents(
    docs=docs,
    embeddings=embeddings,
    hide_annotations=True,
    topics=[0, 1, 2, 3, 4, 5,6, 7, 8, 9],
    height=600,
    width=1000
)


#%%

from gensim.corpora import Dictionary

analyzer = tfidf_vectorizer.build_analyzer()

tokenized_texts = [
    analyzer(doc)
    for doc in df.lemmatized_text
]


dictionary = Dictionary(tokenized_texts)

# Optional but recommended:
dictionary.filter_extremes(no_below=5, no_above=0.95)

# %%
from gensim.models.coherencemodel import CoherenceModel

# 1. Création du corpus Gensim (nécessaire uniquement pour la métrique 'u_mass')
# On utilise ton 'dictionary' et tes 'tokenized_texts' existants
corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

# 2. Initialisation des variables
feature_names = tfidf_vectorizer.get_feature_names_out()
topic_range = [5, 10, 15, 20, 30]
coherence_metrics = ['u_mass', 'c_v', 'c_uci', 'c_npmi']

# Initialisation du dictionnaire de résultats
results = {metrics: {} for metrics in coherence_metrics}

def get_topics(model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        # Get indices of top words (descending order)
        top_indices = np.argsort(-topic)[:n_top_words]
        # Map indices to words
        top_words = [feature_names[i] for i in top_indices]
        topics.append(top_words)
    return topics

feature_names = tfidf_vectorizer.get_feature_names_out()

# for each number of topics, we will compute the 4 metrics
topic_range = [5, 10, 15, 20, 30]
coherence_metrics = ['u_mass', 'c_v', 'c_uci', 'c_npmi']
for metrics in coherence_metrics:
    results[metrics] = {}


for k in topic_range:
    # fit the model with k topics
    nmf = NMF(
        n_components=k,
        random_state=42
    )

    W = nmf.fit_transform(X_tfidf)
    topics = get_topics(nmf, feature_names, 10)
    for met in coherence_metrics:

        coherence_model = CoherenceModel(
            topics=topics,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence=met
        )

        coherence_score = coherence_model.get_coherence()
        results[met][k] =  coherence_score
        print(f"NMF coherence for {k} topic and metrics {met}: {coherence_score}")

#%%
import numpy as np

plt.figure(figsize=(10, 6))

for met in coherence_metrics:
    ks = sorted(results[met].keys())
    scores = np.array([results[met][k] for k in ks])

    # Min-max normalize per metric
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min())

    plt.plot(ks, norm_scores, marker='o', label=met)

plt.xlabel("Number of Topics (k)")
plt.ylabel("Normalized Coherence Score")
plt.title("Normalized Coherence Comparison")
plt.legend()
plt.grid(True)
plt.show()