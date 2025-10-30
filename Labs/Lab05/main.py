import wikipediaapi
from preprocessing_filters import *

wiki_wiki = wikipediaapi.Wikipedia(user_agent='MyProjectName (merlin@example.com)', language='en')


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
t = text[:-1]

t = text_lowercase(t)
t = remove_numbers(t)
t = remove_punctuation(t)
t = remove_whitespace(t)
t = remove_stopwords(t)
t = stem_words(t)

