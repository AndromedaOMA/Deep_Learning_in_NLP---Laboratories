from nltk.corpus import wordnet


def syn_and_ant(word: str):
    synonyms = []
    antonyms = []
    definitions = []

    for syn in wordnet.synsets(word):
        definitions.append(syn.definition())
        for l in syn.lemmas():
            synonyms.append(l.name())

            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    print('Syn', set(synonyms))
    print('Ant', set(antonyms))
    print('DEF', set(definitions))


def get_similarity(word1, word2):
    sl1 = wordnet.synsets(word1)
    sl2 = wordnet.synsets(word2)
    similarities = []
    weights = []
    # print('sl1 size = ', len(sl1), 'sl2 size = ', len(sl2)) # these are always different
    for x in sl1:
        for y in sl2:
            # print(x, y, x.wup_similarity(y))
            sim = x.wup_similarity(y)
            if sim is not None:
                x_weight = sum(lemma.count() for lemma in x.lemmas())
                y_weight = sum(lemma.count() for lemma in y.lemmas())
                weight = x_weight + y_weight
                weights.append(weight)
                similarities.append(sim * weight)
    if not similarities:
        return 0
    if weights == 0:
        weights = 1
    return int(sum(similarities)/sum(weights) * 1000)


if __name__ == '__main__':
    print(get_similarity('Women', 'Car'))
