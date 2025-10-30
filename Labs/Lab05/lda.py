import os

import gensim.corpora as corpora
from pprint import pprint
import gensim
from preprocessing_filters import remove_stopwords

import pyLDAvis.gensim
import pickle
import pyLDAvis


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

    return lda_model, corpus, id2word


def visualize(lda_model, corpus, id2word, num_topics):
    LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_' + str(num_topics))

    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_' + str(num_topics) + '.html')
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)

    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)

    pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_' + str(num_topics) + '.html')

    LDAvis_prepared
