import wikipediaapi
from lda import LDA, visualize
from preprocessing_filters import *

wiki_wiki = wikipediaapi.Wikipedia(user_agent='Lab05', language='en')


def read_content(page: str) -> str:
    p_wiki = wiki_wiki.page(page)
    return p_wiki.text


def get_category_articles(categorymembers, size=15, level=0, max_level=1):
    articles = []

    for c in categorymembers.values():
        if c.ns == wikipediaapi.Namespace.MAIN:
            articles.append(c.title)
            if len(articles) >= size:
                return articles

        elif c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
            sub_articles = get_category_articles(c.categorymembers, size - len(articles), level + 1, max_level)
            articles.extend(sub_articles)
            if len(articles) >= size:
                return articles

    return articles


if __name__ == '__main__':
    categories = ['Science', 'Music', 'Cooking']
    titles = []
    for c in categories:
        current_category = wiki_wiki.page(f"Category:{c}")
        cat = get_category_articles(current_category.categorymembers, 6)
        titles.extend(cat[1:])

    # for title in titles:
    #     print("==================================")
    #     print(read_content(title))

    t = titles[0]
    text = read_content(t)
    filtered_text = text[:-1]

    # 1. Preprocessing the text
    filtered_text = text_lowercase(filtered_text)
    filtered_text = remove_numbers(filtered_text)
    filtered_text = remove_punctuation(filtered_text)
    filtered_text = remove_whitespace(filtered_text)

    # Compare original and filtered text:
    # with open("./data/original_text", "a") as f:
    #     f.write(text)
    # with open("./data/filtered_text", "a") as f:
    #     f.write(filtered_text)
    #
    # with open(r"./data/original_text", 'r') as f1, open(r"./data/filtered_text", 'r') as f2:
    #     line_num = 0
    #     for line1, line2 in zip(f1, f2):
    #         line_num += 1
    #         if line1 != line2:
    #             print(f"Line {line_num}:")
    #             print(f"\tFile 1: {line1.strip()}")
    #             print(f"\tFile 2: {line2.strip()}")


    # 2. Latent Semantic Analysis with SVD

    # 3. Non-negative matrix factorization

    # 4. LDA
    lda_model, corpus, id2word = LDA(filtered_text, num_topics=3)
    visualize(lda_model, corpus, id2word, num_topics=3)
