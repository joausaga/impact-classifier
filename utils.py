from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import unicodedata


def remove_non_ascii(words):
    """Remove non-ASCII characters from words"""
    words = get_list(words)
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters of words to lowercase"""
    words = get_list(words)
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from words"""
    words = get_list(words)
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_extra_spaces(words):
    words = get_list(words)
    new_words = []
    for word in words:
        word_clean = ' '.join(word.split())
        new_words.append(word_clean)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    words = get_list(words)
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def get_list(text):
    if not isinstance(text, list):
        return word_tokenize(text)
    else:
        return text


def normalize(words):
    words = get_list(words)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_extra_spaces(words)
    words = remove_stopwords(words)
    return words


def lemmatize_words(words, pos='v'):
    """Lemmatize verbs in list of tokenized words"""
    words = get_list(words)
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos=pos)
        lemmas.append(lemma)
    return lemmas