from gensim import corpora
from gensim.models import LsiModel, TfidfModel

from vectorization import preprocess_text
from wiki_methods import read_content
from preprocessing_filters import remove_stopwords
import numpy as np
import pyLDAvis


# https://www.datacamp.com/tutorial/discovering-hidden-topics-python
def create_gensim_lsa_model(titles, number_of_topics, words, preprocessing=True):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    text_corpus = []
    for title in titles:
        text = read_content(title)
        if preprocessing:
            tokens = preprocess_text(text)
            if isinstance(tokens, str):
                tokens = tokens.split()
        text_corpus.append(tokens)

    if not text_corpus:
        raise ValueError("No valid texts to process — check input titles.")

    print(f"\n[INFO] Collected {len(text_corpus)} documents for LDA.\n")

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(text_corpus)
    corpus = [dictionary.doc2bow(text) for text in text_corpus]

    # generate LSA model (https://radimrehurek.com/gensim/models/lsimodel.html)
    lsa_model = LsiModel(corpus, num_topics=number_of_topics, id2word=dictionary)  # train model
    print(f'lsa_model: {lsa_model}')
    print(f'lsa_model.print_topics: {lsa_model.print_topics(num_topics=number_of_topics, num_words=words)}')
    return lsa_model


def create_adapted_gensim_lsa_model(corpus, dictionary, number_of_topics, words):
    lsa_model = LsiModel(corpus, num_topics=number_of_topics, id2word=dictionary)
    new_document_lsi = lsa_model[corpus]

    print(f'lsa_model: {lsa_model}')
    print(f'lsa_model.print_topics: {lsa_model.print_topics(num_topics=number_of_topics, num_words=words)}')

    return lsa_model, new_document_lsi


def visualize_lsa(lsa_model, corpus_bow, dictionary):
    n_topics = lsa_model.num_topics
    vocab = [dictionary[i] for i in range(len(dictionary))]

    corpus_bow_list = list(corpus_bow)
    W = np.zeros((len(corpus_bow_list), n_topics), dtype=float)
    for i, doc in enumerate(corpus_bow_list):
        topic_dist = lsa_model[doc]
        for topic_id, weight in topic_dist:
            W[i, topic_id] = abs(weight)

    Wn = W / (W.sum(axis=1, keepdims=True) + 1e-12)

    H = np.zeros((n_topics, len(dictionary)), dtype=float)
    for topic_id in range(n_topics):
        topic = lsa_model.show_topic(topic_id, topn=len(dictionary))
        for word, weight in topic:
            word_id = dictionary.token2id.get(word)
            if word_id is not None:
                H[topic_id, word_id] = abs(weight)

    Hn = H / (H.sum(axis=1, keepdims=True) + 1e-12)

    doc_lengths = np.array([int(sum(count for _, count in doc)) for doc in corpus_bow], dtype=int)

    term_frequency = np.zeros(len(dictionary), dtype=int)
    for doc in corpus_bow:
        for word_id, count in doc:
            term_frequency[word_id] += int(count)

    vis = pyLDAvis.prepare(
        topic_term_dists=Hn,
        doc_topic_dists=Wn,
        doc_lengths=doc_lengths,
        vocab=vocab,
        term_frequency=term_frequency,
        mds='mmds',
        sort_topics=False
    )

    return vis
