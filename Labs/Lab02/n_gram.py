# corpus link https://github.com/lmidriganciochina/romaniancorpus/blob/master/Corpus_Data/Written%20Language/Manuale%20-%20Textbooks/Gramatica%20Rom%C3%A2n%C4%83.txt
import re
import math
from collections import Counter


def read_corpus(filename):
    with open(filename, encoding="utf-8") as f:
        text = f.read().lower()

    text = (text
            .replace("ş", "ș")
            .replace("ţ", "ț")
            .replace("Ş", "Ș")
            .replace("Ţ", "Ț")
            )

    text = re.sub(r"(?<![a-zăâîșț])-|-(?![a-zăâîșț])", " ", text)
    text = re.sub(r"[^a-zăâîșț\-\s\.!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


corpus = read_corpus("corpus.txt")


def sentence_tokenize(text):
    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


s_token = sentence_tokenize(corpus)


def add_sentence_markers(sentences):
    return [["<s>"] + s.split() + ["</s>"] for s in sentences]


s_with_markers = add_sentence_markers(s_token)


def generate_ngram_contexts(sentences_with_markers, n):
    pairs = []
    for sent in sentences_with_markers:
        for i in range(n - 1, len(sent)):
            history = tuple(sent[i - (n - 1):i]) if n > 1 else tuple()
            next_tok = sent[i]
            pairs.append((history, next_tok))
    return pairs


def train_ngram_with_context(sentences_with_markers, n):
    pairs = generate_ngram_contexts(sentences_with_markers, n)
    ngram_counts = Counter()
    history_counts = Counter()
    for h, w in pairs:
        ngram_counts[h + (w,)] += 1
        history_counts[h] += 1

    vocab = set(tok for sent in sentences_with_markers for tok in sent)
    V = len(vocab)

    def P(next_tok, history):
        h = tuple(history)
        return (ngram_counts[h + (next_tok,)] + 1) / (history_counts[h] + V)

    return P, ngram_counts, history_counts, vocab


def sentence_prob(sentence_str, n, P):
    toks = ["<s>"] + sentence_str.split() + ["</s>"]
    total_prob = 1.0
    for i in range(n - 1, len(toks)):
        history = tuple(toks[i - (n - 1):i]) if n > 1 else tuple()
        w = toks[i]
        p = P(w, history)
        total_prob *= p
    return total_prob


def perplexity(sentences_raw, n, P):
    N = 0
    S = 0.0
    for s in sentences_raw:
        toks = ["<s>"] + s.split() + ["</s>"]
        for i in range(n - 1, len(toks)):
            history = tuple(toks[i - (n - 1):i]) if n > 1 else tuple()
            w = toks[i]
            p = P(w, history)
            S += -math.log(p)
            N += 1
    return math.exp(S / max(N, 1))


n = 3
P, ngram_counts, history_counts, vocab = train_ngram_with_context(s_with_markers, n)

print("P(prima propoziție) =", sentence_prob(s_token[0], n, P))
ppl = perplexity(s_token, 3, P)
print(f"Perplexity (N={n}): {ppl:.3f}")


def normalize_line(line: str) -> str:
    line = line.lower()
    line = re.sub(r"(?<![a-zăâîșț])-|-(?![a-zăâîșț])", " ", line)
    line = re.sub(r"[^a-zăâîșț\-\s\.!?]", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def prob_of_sentence(sentence_str: str, n: int, P) -> float:
    sent_norm = normalize_line(sentence_str)
    return sentence_prob(sent_norm, n, P)


s_noua = "cinci minute"
p = prob_of_sentence(s_noua, n, P)
print(f"P('{s_noua}') = {p:.12e}")
