import string
import logging

import numpy as n
from bugs.models import BugCategory, BugReport
from django.db.models import Count
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora

from .models import Frequency, FrequencyDetail, FrequencyTitle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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


def all_issues(test1):
    documents = []
    for bug in BugReport.objects.all():
        documents.append(set(tokenize(bug.title)))
    texts = [[word for word in ("").join(document).lower().split()] for document in documents]
    dictionary = corpora.Dictionary(texts)
    dictionary.save('/tmp/deerwester.dict')  # store the dictionary, for future reference
    print (dictionary)
    return dictionary



