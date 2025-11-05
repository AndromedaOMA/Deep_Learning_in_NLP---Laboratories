import os
import gensim.corpora as corpora
import gensim
from vectorization import preprocess_text
from wiki_methods import read_content
import pyLDAvis.gensim
import pickle
import pyLDAvis


def LDA(titles, preprocessing=True, num_topics=3):
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
    id2word = corpora.Dictionary(text_corpus)
    corpus = [id2word.doc2bow(text) for text in text_corpus]

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
