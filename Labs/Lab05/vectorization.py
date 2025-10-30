from preprocessing_filters import *
from wiki_methods import *


def BoW(titles):
    for title in titles:
        text = read_content(title)
        filtered_text = text[:-1]

        # 1. Preprocessing the text
        filtered_text = text_lowercase(filtered_text)
        filtered_text = remove_numbers(filtered_text)
        filtered_text = remove_punctuation(filtered_text)
        filtered_text = remove_whitespace(filtered_text)

        list_of_words = remove_stopwords(filtered_text)
        word2count = word_freq(list_of_words)
        print(word2count)