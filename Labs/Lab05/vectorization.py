from preprocessing_filters import *
from wiki_methods import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


def sklearn_BoW(titles):
    corpus = []
    for title in titles:
        text = read_content(title)
        corpus.append(text)
    # Create a CountVectorizer Object
    vectorizer = CountVectorizer()
    # Fit and transform the corpus
    X = vectorizer.fit_transform(corpus)
    # Print the generated vocabulary
    print("Vocabulary:", vectorizer.get_feature_names_out())
    # Print the Bag-of-Words matrix
    print("BoW Representation:")
    print(X.toarray())


def vocab_generator(titles, no_of_tokens=20):
    word2count = {}
    for title in titles:
        text = read_content(title)
        filtered_text = text[:-1]

        # Preprocessing the text
        filtered_text = text_lowercase(filtered_text)
        filtered_text = remove_numbers(filtered_text)
        filtered_text = remove_punctuation(filtered_text)
        filtered_text = remove_whitespace(filtered_text)

        # vocabulary
        list_of_lemma_words = lemma_words(filtered_text)
        # list_of_stemmed_words = stem_words(list_of_lemma_words)
        # list_of_tokens = remove_stopwords(list_of_stemmed_words)
        list_of_tokens = remove_stopwords(list_of_lemma_words)
        fr = word_freq(list_of_tokens)
        word2count.update(fr)

    vocab = [word for word, _ in sorted(word2count.items(), key=lambda x: x[1], reverse=True)[:no_of_tokens]]
    # vocab = {k: v for k, v in sorted(word2count.items(), key=lambda item: item[1], reverse=True)}
    print(vocab)
    return vocab


def BoW(titles, no_of_tokens=20):
    vocab = vocab_generator(titles, no_of_tokens)
    bow = []
    for title in titles:
        current_bow = np.zeros(len(vocab))
        text = read_content(title)

        filtered_text = text_lowercase(text)
        filtered_text = remove_numbers(filtered_text)
        filtered_text = remove_punctuation(filtered_text)
        filtered_text = remove_whitespace(filtered_text)
        list_of_lemma_words = lemma_words(filtered_text)
        tokens = remove_stopwords(list_of_lemma_words)

        for i, word in enumerate(vocab):
            if word in tokens:
                current_bow[i] = 1
        bow.append(current_bow)

    bow = np.asarray(bow, dtype=int)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        bow,
        cmap='RdYlGn',
        cbar=False,
        annot=True,
        fmt="d",
        xticklabels=vocab,
        yticklabels=[f"Title {i + 1}" for i in range(len(titles))]
    )
    plt.title('Bag of Words Matrix')
    plt.xlabel('Frequent Words')
    plt.ylabel('Titles')
    plt.tight_layout()
    plt.show()

    return bow
