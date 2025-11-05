from gensim import corpora

from preprocessing_filters import *
from wiki_methods import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter


def preprocess_text(text):
    t = text_lowercase(text)
    t = remove_numbers(t)
    t = remove_punctuation(t)
    t = remove_whitespace(t)
    toks = lemma_words(t)
    toks = remove_stopwords(toks)
    toks = [w for w in toks if w.isalpha() and len(w) > 3]
    return " ".join(toks)


def sklearn_BoW(titles, preprocessing=True):
    corpus = []
    for title in titles:
        text = read_content(title)
        if preprocessing:
            text = preprocess_text(text)
        corpus.append(text)
    # Create a CountVectorizer Object
    vectorizer = CountVectorizer(
        token_pattern=None if preprocessing else r"(?u)\b\w\w+\b",
        tokenizer=str.split if preprocessing else None,
        binary=True
    )
    # Fit and transform the corpus
    X = vectorizer.fit_transform(corpus)
    # Print the generated vocabulary
    print("Vocabulary:", len(vectorizer.get_feature_names_out()))
    # Print the Bag-of-Words matrix
    print(f"BoW Representation: {X.toarray()}")


def sklearn_tf_idf(titles, preprocessing=True):
    corpus = []
    for title in titles:
        text = read_content(title)
        if preprocessing:
            text = preprocess_text(text)
        corpus.append(text)
    vectorizer = TfidfVectorizer(
        token_pattern=None if preprocessing else r"(?u)\b\w\w+\b",
        tokenizer=str.split if preprocessing else None
    )
    X = vectorizer.fit_transform(corpus)
    # print("Vocabulary:", vectorizer.get_feature_names_out())
    # Print the TF-IDF matrix
    # print(f"TFIDF Representation: {X.toarray()}")
    return X, vectorizer


def vocab_generator(titles, no_of_tokens=20):
    word2count = Counter()
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
        fr = Counter(list_of_tokens)
        word2count.update(fr)

    vocab = [word for word, _ in sorted(word2count.items(), key=lambda x: x[1], reverse=True)[:no_of_tokens]]
    # vocab = {k: v for k, v in sorted(word2count.items(), key=lambda item: item[1], reverse=True)}
    print(vocab)
    return vocab


def BoW(titles, no_of_tokens=20):
    vocab = vocab_generator(titles, no_of_tokens)
    manual_bow = []
    texts = []
    for title in titles:
        current_bow = np.zeros(len(vocab))
        text = read_content(title)

        filtered_text = text_lowercase(text)
        filtered_text = remove_numbers(filtered_text)
        filtered_text = remove_punctuation(filtered_text)
        filtered_text = remove_whitespace(filtered_text)
        list_of_lemma_words = lemma_words(filtered_text)
        tokens = remove_stopwords(list_of_lemma_words)
        texts.append(tokens)

        for i, word in enumerate(vocab):
            if word in tokens:
                current_bow[i] = 1
        manual_bow.append(current_bow)

    # BoW format
    dictionary = corpora.Dictionary(texts)
    bow_corpus = [dictionary.doc2bow(text) for text in texts]

    # visualization
    bow = np.asarray(manual_bow, dtype=int)

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

    # manual_bow, vocab (manual implementation)
    # bow_corpus, dictionary (BoW format)
    return manual_bow, vocab, bow_corpus, dictionary, texts
