from gensim import corpora
from gensim.models import LsiModel
from preprocessing_filters import remove_stopwords


# https://www.datacamp.com/tutorial/discovering-hidden-topics-python
def create_gensim_lsa_model(filtered_text, number_of_topics, words):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    # removing stop words
    tokens_wout_stopwords = remove_stopwords(filtered_text)
    # Create Dictionary
    dictionary = corpora.Dictionary([tokens_wout_stopwords])
    # Create Corpus
    texts = [tokens_wout_stopwords]
    # Term Document Frequency
    corpus = [dictionary.doc2bow(text) for text in texts]

    # generate LSA model (https://radimrehurek.com/gensim/models/lsimodel.html)
    lsa_model = LsiModel(corpus, num_topics=number_of_topics, id2word=dictionary)  # train model
    print(lsa_model.print_topics(num_topics=number_of_topics, num_words=words))
    return lsa_model
