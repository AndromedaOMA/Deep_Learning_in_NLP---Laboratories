from gensim import corpora
from gensim.models import LsiModel, TfidfModel

from vectorization import preprocess_text
from wiki_methods import read_content
from preprocessing_filters import remove_stopwords


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


def create_adapted_gensim_lsa_model(bow_corpus, dictionary, number_of_topics, words, bow=True):
    # Create dictionary and corpus
    dictionary = dictionary
    bow_corpus_for_lsa = bow_corpus

    if bow:
        # generate LSA model (https://radimrehurek.com/gensim/models/lsimodel.html)
        lsa_model = LsiModel(bow_corpus_for_lsa, num_topics=number_of_topics, id2word=dictionary)
        # 2. Transform a BoW document into the LSI topic space
        new_document_lsi = lsa_model[bow_corpus[0]]
    else:
        tfidf_model = TfidfModel(bow_corpus)
        tfidf_corpus_for_lsa = tfidf_model[bow_corpus]
        lsa_model = LsiModel(tfidf_corpus_for_lsa, num_topics=number_of_topics, id2word=dictionary)
        new_document_lsi = lsa_model[tfidf_corpus_for_lsa[0]]

    print(f'lsa_model: {lsa_model}')
    print(f'lsa_model.print_topics: {lsa_model.print_topics(num_topics=number_of_topics, num_words=words)}')

    return lsa_model, new_document_lsi
