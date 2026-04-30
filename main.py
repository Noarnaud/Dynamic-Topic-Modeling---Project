"""
main.py — Orchestrateur principal du projet Dynamic Topic Modeling.

Pipeline complète :
  1. Chargement et prétraitement des données (toutes les années)
  2. Analyse exploratoire (statistiques + graphiques du corpus)
  3. LDA baseline (entraînement + visualisation)
  4. BERTopic (statique + dynamique + topics_over_time)
  5. Comparaison de cohérence (LDA vs BERTopic)
  6. Sauvegarde de tous les résultats

Usage :
  python main.py                           # Pipeline complète (4 années, 10 topics)
  python main.py --years 81                # Une seule année
  python main.py --years 73 78 81 88       # Toutes les années
  python main.py --n-topics 15 --no-cache  # 15 topics, sans cache
  python main.py --skip-lda                # Seulement BERTopic
  python main.py --skip-bert               # Seulement LDA
"""

import argparse
import os
import sys

# Ajouter le répertoire du projet au path pour les imports
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from src.config import AVAILABLE_YEARS, N_TOPICS, OUTPUT_DIR
from src.data_loader import load_all_years
from src.lda_model import run_lda_pipeline, get_topics, plot_top_words, interactive_vis
from src.bert_model import (
    get_or_create_embeddings, train_bertopic, train_dynamic_bertopic,
    get_bertopic_topics, save_bertopic_visualizations
)
from src.coherence import (
    prepare_gensim_data, compare_models,
    plot_coherence_comparison, coherence_vs_k, plot_coherence_vs_k
)
from src.visualization import (
    plot_corpus_distribution, plot_party_distribution,
    plot_text_length_distribution, plot_topic_heatmap, save_figure
)


def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic Topic Modeling — Archelec")
    parser.add_argument('--years', nargs='+', type=int, default=AVAILABLE_YEARS,
                        help=f"Années à analyser (défaut: {AVAILABLE_YEARS})")
    parser.add_argument('--n-topics', type=int, default=N_TOPICS,
                        help=f"Nombre de topics (défaut: {N_TOPICS})")
    parser.add_argument('--no-cache', action='store_true',
                        help="Désactiver le cache (recalculer tout)")
    parser.add_argument('--skip-lda', action='store_true',
                        help="Sauter l'analyse LDA")
    parser.add_argument('--skip-bert', action='store_true',
                        help="Sauter l'analyse BERTopic")
    parser.add_argument('--skip-coherence', action='store_true',
                        help="Sauter la comparaison de cohérence")
    return parser.parse_args()


def main():
    args = parse_args()
    use_cache = not args.no_cache

    print("=" * 60)
    print("   DYNAMIC TOPIC MODELING — ARCHELEC CORPUS")
    print("=" * 60)
    print(f"  Années       : {args.years}")
    print(f"  Topics       : {args.n_topics}")
    print(f"  Cache        : {'✅' if use_cache else '❌'}")
    print(f"  LDA          : {'✅' if not args.skip_lda else '⏭️ skip'}")
    print(f"  BERTopic     : {'✅' if not args.skip_bert else '⏭️ skip'}")
    print(f"  Cohérence    : {'✅' if not args.skip_coherence else '⏭️ skip'}")
    print("=" * 60)

    # ==============================================================
    # CHARGEMENT ET PRÉTRAITEMENT
    # ==============================================================

    df_global = load_all_years(years=args.years, use_cache=use_cache)
    print(f"\n Corpus total : {len(df_global)} documents")

    # ==============================================================
    # ANALYSE EXPLORATOIRE
    # ==============================================================

    fig = plot_corpus_distribution(df_global)
    save_figure(fig, "01_corpus_distribution.png")

    fig = plot_party_distribution(df_global)
    save_figure(fig, "02_party_distribution.png")

    fig = plot_text_length_distribution(df_global)
    save_figure(fig, "03_text_length_distribution.png")

    # Variables pour la comparaison de cohérence
    lda_topics_for_coherence = None
    bert_topics_for_coherence = None

    # ==============================================================
    # LDA BASELINE
    # ==============================================================
    if not args.skip_lda:
        # LDA sur le corpus global
        lda_model, tf, tf_vectorizer = run_lda_pipeline(df_global, n_topics=args.n_topics)

        # Visualisation des top words
        fig = plot_top_words(lda_model, tf_vectorizer, title=f"LDA — {args.n_topics} Topics (toutes années)")
        save_figure(fig, "04_lda_top_words.png")

        # pyLDAvis
        lda_html = os.path.join(OUTPUT_DIR, "05_lda_interactive.html")
        interactive_vis(lda_model, tf, tf_vectorizer, output_file=lda_html)

        # Extraction des topics pour la comparaison
        lda_topics_for_coherence = get_topics(lda_model, tf_vectorizer)

        # LDA par année 
        for year in args.years:
            df_year = df_global[df_global['annee'] == year]
            if len(df_year) > 50:
                print(f"\n  --- LDA 19{year:02d} ({len(df_year)} docs) ---")
                lda_y, tf_y, vec_y = run_lda_pipeline(df_year, n_topics=args.n_topics)
                fig = plot_top_words(lda_y, vec_y, title=f"LDA — 19{year:02d}")
                save_figure(fig, f"04_lda_top_words_{year:02d}.png")

    # ==============================================================
    # BERTOPIC
    # ==============================================================
    if not args.skip_bert:

        # Embeddings (avec cache)
        cache_name = f"embeddings_{'_'.join(map(str, args.years))}"
        embeddings = get_or_create_embeddings(df_global, name=cache_name, use_cache=use_cache)

        # Modèle dynamique (global + topics_over_time)
        topic_model, topics, probs, topics_over_time = train_dynamic_bertopic(
            df_global, embeddings, n_topics=args.n_topics
        )

        # Infos sur les topics
        topic_info = topic_model.get_topic_info()
        print("\n Topics découverts :")
        print(topic_info.head(args.n_topics + 1).to_string(index=False))

        # Sauvegarde des visualisations BERTopic
        print("\n Génération des visualisations BERTopic...")
        save_bertopic_visualizations(
            topic_model,
            topics_over_time=topics_over_time,
            docs=df_global['lemmatized_text'].tolist(),
            embeddings=embeddings,
            prefix="06_bertopic"
        )

        # Heatmap topic × année
        fig = plot_topic_heatmap(topic_model, df_global, topics)
        save_figure(fig, "07_topic_heatmap.png")

        # Topics pour la comparaison
        bert_topics_for_coherence = get_bertopic_topics(topic_model)

        # Sauvegarder le topic_info en CSV
        topic_info.to_csv(os.path.join(OUTPUT_DIR, "08_topic_info.csv"), index=False)
        print(f"Topic info : {os.path.join(OUTPUT_DIR, '08_topic_info.csv')}")

    # ==============================================================
    # COMPARAISON DE COHÉRENCE
    # ==============================================================
    if not args.skip_coherence and lda_topics_for_coherence and bert_topics_for_coherence:

        tokenized_texts, dictionary = prepare_gensim_data(df_global)

        results_df = compare_models(
            lda_topics_for_coherence,
            bert_topics_for_coherence,
            tokenized_texts,
            dictionary
        )

        print(results_df.to_string(index=False))

        # Graphique comparatif
        fig = plot_coherence_comparison(results_df)
        save_figure(fig, "09_coherence_comparison.png")

        # Sauvegarder les résultats en CSV
        results_df.to_csv(os.path.join(OUTPUT_DIR, "09_coherence_results.csv"), index=False)

    # ==============================================================
    # RÉSUMÉ FINAL
    # ==============================================================
    print("\n\n" + "=" * 60)
    print("  ✅ PIPELINE TERMINÉE")
    print("=" * 60)
    print(f"  📁 Tous les résultats dans : {OUTPUT_DIR}")

    output_files = os.listdir(OUTPUT_DIR)
    output_files = [f for f in output_files if not f.startswith('.') and f != 'cache']
    print(f"  📄 {len(output_files)} fichiers générés :")
    for f in sorted(output_files):
        print(f"     • {f}")


if __name__ == "__main__":
    main()
