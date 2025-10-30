from vectorization import *
from lda import LDA, visualize
from wiki_methods import *


if __name__ == '__main__':
    categories = ['Science', 'Music', 'Cooking']
    titles = []
    for c in categories:
        current_category = wiki_wiki.page(f"Category:{c}")
        cat = get_category_articles(current_category.categorymembers, 6)
        titles.extend(cat[1:])

    # 1. BoW & TF-IDF
    BoW(titles)
    # sklearn_BoW(titles)

    # 2. Latent Semantic Analysis with SVD

    # 3. Non-negative matrix factorization

    # 4. LDA (DONE)
    # lda_model, corpus, id2word = LDA(filtered_text, num_topics=3)
    # visualize(lda_model, corpus, id2word, num_topics=3)
