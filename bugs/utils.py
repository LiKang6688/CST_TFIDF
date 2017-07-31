import string

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def tokenize(text):
    """Returns a list of stemmed and stopwords-removed tokens of the
    given text.

    Parameters
    ----------
    text : string
        The text to tokenize
    """
    lowers = text.lower()
    punct_free = lowers.translate(str.maketrans({key: None for key in string.punctuation}))
    tokens = word_tokenize(punct_free)
    porter = PorterStemmer()
    stemmed = [porter.stem(w) for w in tokens]
    tokens = [w for w in stemmed if w not in stopwords.words('english')]
    return tokens
