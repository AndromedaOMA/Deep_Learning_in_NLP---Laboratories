from transformers import AutoTokenizer
from collections import defaultdict, Counter
import math

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

tokenizer = AutoTokenizer.from_pretrained("gpt2")

word_freqs = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, _ in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

alphabet = []
for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

vocab = ["<|endoftext|>"] + alphabet.copy()
splits = {word: [c for c in word] for word in word_freqs.keys()}


def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def merge_pair(a, b, splits):
    for word in list(word_freqs.keys()):
        s = splits[word]
        if len(s) == 1:
            continue
        i = 0
        while i < len(s) - 1:
            if s[i] == a and s[i + 1] == b:
                s = s[:i] + [a + b] + s[i + 2:]
            else:
                i += 1
        splits[word] = s
    return splits


vocab_size = 50
merges_list = []

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    if not pair_freqs:
        break
    best_pair, max_freq = None, -1
    for pair, freq in pair_freqs.items():
        if freq > max_freq:
            best_pair, max_freq = pair, freq
    if max_freq <= 0 or best_pair is None:
        break

    a, b = best_pair
    ab = a + b
    splits = merge_pair(a, b, splits)
    merges_list.append((a, b, ab))
    vocab.append(ab)

print("\nAll merges")
print(merges_list)
print("\nSize vocab:", len(vocab))


def encode_word(word):
    pieces = [c for c in word]
    for (a, b, ab) in merges_list:
        i = 0
        while i < len(pieces) - 1:
            if pieces[i] == a and pieces[i + 1] == b:
                pieces = pieces[:i] + [ab] + pieces[i + 2:]
            else:
                i += 1
    return pieces


def encode_text(text):
    toks = []
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    words = [w for (w, _) in words_with_offsets]
    for w in words:
        toks.extend(encode_word(w))
    return toks


print("\nBPE ENCODE")
print(encode_text(corpus[0]))
print(encode_text("This is not a token"))

BOS, EOS = "<s>", "</s>"


def train_bigram_lm_bpe(corpus_texts):
    sequences = []
    for t in corpus_texts:
        sequences.append([BOS] + encode_text(t) + [EOS])

    uni = Counter()
    bi = Counter()
    for seq in sequences:
        for i, tok in enumerate(seq):
            uni[tok] += 1
            if i > 0:
                bi[(seq[i - 1], tok)] += 1

    V = len(uni)

    def P(next_tok, prev_tok):
        return (bi[(prev_tok, next_tok)] + 1.0) / (uni[prev_tok] + V)

    return P, uni, bi, V


P, uni, bi, V = train_bigram_lm_bpe(corpus)


def perplexity(texts):
    total_prob = 1.0
    N = 0
    for t in texts:
        toks = [BOS] + encode_text(t) + [EOS]
        for i in range(1, len(toks)):
            total_prob *= P(toks[i], toks[i - 1])
        N += (len(toks) - 1)
    return (1 / total_prob) ** (1 / N)


print("\nPerplexity")
print(perplexity(corpus))
