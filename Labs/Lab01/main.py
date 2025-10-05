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


if __name__ == '__main__':
    sl1 = wordnet.synsets("good")
    sl2 = wordnet.synsets("fantastic")
    for x in sl1:
        for y in sl2:
            print(x, y, x.wup_similarity(y))
