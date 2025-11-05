from sklearn.decomposition import NMF
import numpy as np
from vectorization import sklearn_tf_idf


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
