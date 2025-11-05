from lsa_with_svd import create_gensim_lsa_model, create_adapted_gensim_lsa_model
from vectorization import *
from lda import LDA, visualize
from wiki_methods import *
from nmf import run_nmf_from_matrix

if __name__ == '__main__':
    categories = ['Science', 'Music', 'Cooking']
    titles = []
    for c in categories:
        current_category = wiki_wiki.page(f"Category:{c}")
        cat = get_category_articles(current_category.categorymembers, 6)
        titles.extend(cat[1:])

    # 1. BoW & TF-IDF (DONE)
    manual_bow, vocab, bow_corpus, dictionary, texts = BoW(titles)
    # sklearn_BoW(titles)
    # X_tfidf, vec = sklearn_tf_idf(titles)

    # 2. Latent Semantic Analysis with SVD (DONE)
    lsa_model, new_document_lsi = create_adapted_gensim_lsa_model(bow_corpus, dictionary, number_of_topics=3, words=5, bow=True)

    # 3. Non-negative matrix factorization
    # W, H, model = run_nmf_from_matrix(X_tfidf, vec)

    # 4. LDA (DONE)
    # lda_model, corpus, id2word = LDA(titles)
    # visualize(lda_model, corpus, id2word, num_topics=3)
