import nltk
import string
import re
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('wordnet')


def text_lowercase(text):
    return text.lower()


def remove_numbers(text):
    return re.sub(r'\d+', '', text)


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def remove_whitespace(text):
    return " ".join(text.split())


def remove_stopwords(list_of_tokens):
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in list_of_tokens if word.lower()
                     not in stop_words]
    return filtered_text


stemmer = PorterStemmer()


def stem_words(list_of_tokens):
    stems = [stemmer.stem(word) for word in list_of_tokens]
    return stems


lemmatizer = WordNetLemmatizer()


def lemma_words(text):
    word_tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word) for word in word_tokens]
    return lemmas


def word_freq(list_of_word):
    word2count = {}
    for word in list_of_word:
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

    return word2count
