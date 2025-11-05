from sklearn.decomposition import NMF
import numpy as np
from vectorization import preprocess_text
import pyLDAvis
from wiki_methods import read_content
from gensim import corpora, models
from gensim.models import Nmf


def run_nmf_from_matrix(X, vectorizer, n_topics=3, topn=30):
    model = NMF(n_components=n_topics, init='nndsvda', random_state=0)
    W = model.fit_transform(X)
    H = model.components_

    terms = vectorizer.get_feature_names_out()
    for k, row in enumerate(H):
        idx = row.argsort()[::-1][:topn]
        print(f"Topic {k + 1}: {'  '.join(terms[i] for i in idx)}")

    Wn = W / (W.sum(axis=1, keepdims=True) + 1e-12)
    print("\nTopic distribution:")
    print(np.round(Wn, 3))

    return W, H, model


def build_gensim_corpus(titles, preprocessing=True):
    docs_tokens = []
    for title in titles:
        txt = read_content(title)
        if preprocessing:
            txt = preprocess_text(txt)
            toks = txt.split()
        else:
            toks = txt.split()
        docs_tokens.append(toks)

    dictionary = corpora.Dictionary(docs_tokens)
    corpus_bow = [dictionary.doc2bow(toks) for toks in docs_tokens]
    tfidf = models.TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]
    return dictionary, corpus_bow, corpus_tfidf


def run_nmf_from_corpus(corpus_tfidf, dictionary, n_topics=3, topn=30, random_state=0):
    nmf = Nmf(
        corpus=corpus_tfidf,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=random_state
    )

    H = nmf.get_topics()

    corpus_tfidf_list = list(corpus_tfidf)
    W = np.zeros((len(corpus_tfidf_list), n_topics), dtype=float)
    for i, doc in enumerate(corpus_tfidf_list):
        for k, w in nmf[doc]:
            W[i, k] = w

    for k in range(n_topics):
        idx = np.argsort(H[k])[::-1][:topn]
        terms = [dictionary[j] for j in idx]
        print(f"Topic {k + 1}: {'  '.join(terms)}")

    Wn = W / (W.sum(axis=1, keepdims=True) + 1e-12)
    Hn = H / (H.sum(axis=1, keepdims=True) + 1e-12)
    print("\nTopic distribution:")
    print(np.round(Wn, 3))

    return nmf, Wn, Hn


def visualize_nmf(Wn, Hn, corpus_bow, dictionary):
    vocab = [dictionary[i] for i in range(len(dictionary))]

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
