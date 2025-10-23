# link: https://medium.com/plain-simple-software/the-stanza-nlp-library-basics-of-text-preprocessing-continued-28a1743318a3
import stanza


nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

sentences = [
    'Flying planes can be dangerous',
    'The parents of the bride and the groom were flying',
    'The groom loves dangerous planes more than the bride'
]

for s in sentences:
    print(f"\n=== {s} ===")
    doc = nlp(s)
    print(*[
        f"id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head - 1].text if word.head > 0 else 'root'}\tdeprel: {word.deprel}"
        for sent in doc.sentences
        for word in sent.words
    ], sep='\n')
