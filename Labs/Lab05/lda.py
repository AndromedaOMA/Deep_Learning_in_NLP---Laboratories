import gensim.corpora as corpora
from pprint import pprint
import gensim
# from multiprocessing import freeze_support
from preprocessing_filters import remove_stopwords


def LDA(filtered_text, num_topics):
    # removing stop words
    tokens_wout_stopwords = remove_stopwords(filtered_text)
    # Create Dictionary
    id2word = corpora.Dictionary([tokens_wout_stopwords])
    # Create Corpus
    texts = [tokens_wout_stopwords]
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    print(f'Corpus sample: {corpus[:1][0][:30]}')

    # Build LDA model
    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics)  # Print the Keyword in the 10 topics

    for idx, topic in lda_model.print_topics(num_words=10):
        print(f"Topic {idx + 1}:\n  {topic}\n")

    return lda_model
