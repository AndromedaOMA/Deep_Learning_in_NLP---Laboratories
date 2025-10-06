from random_word import RandomWords


def generate_words(on_event="Start"):
    r = RandomWords()

    if on_event == "Start":
        list_words = []
        while len(list_words) < 8:
            word = r.get_random_word()
            if word and word not in list_words:
                list_words.append(word)
        return list_words

    else:
        return r.get_random_word()
