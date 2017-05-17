import string
import gensim

import numpy as n
from bugs.models import BugCategory, BugReport
from django.db.models import Count
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from .models import Frequency, FrequencyDetail, FrequencyTitle


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


# The term frequencies (TF)

def tf(bug, term):
    """Return the term frequency of a term in a bug report.
    This is based on both title and description fields.
    """
    qs = Frequency.objects.filter(bug=bug, term=term)
    if qs.count() > 0:
        return qs.first().freq
    else:
        return 0


def tf_title(bug, term):
    """Return the term frequency of a term in a bug report.
    This is based on title field.
    """
    qs = FrequencyTitle.objects.filter(bug=bug, term=term)
    if qs.count() > 0:
        return qs.first().freq
    else:
        return 0


def tf_detail(bug, term):
    """Return the term frequency of a term in a bug report.
    This is based on description field.
    """
    qs = FrequencyDetail.objects.filter(bug=bug, term=term)
    if qs.count() > 0:
        return qs.first().freq
    else:
        return 0


# The inverse document frequencies (IDF)

def idf(term):
    """Returns the IDF of a term based on title + description.
    """
    # da = BugReport.objects.count()
    # dt = Frequency.objects.filter(term=term).count()
    # return n.log2(float(da) / (1 + dt))
    da = Frequency.objects.filter(term=term)
    if da.count() > 0:
        return da.first().idf
    else:
        return 0


def idf_title(term):
    """Returns the IDF of a term based on title.
    """
    # da = BugReport.objects.count()
    # dt = FrequencyTitle.objects.filter(term=term).count()
    # print("da ", ds[0].idf)
    # return n.log2(float(da) / (1 + dt))
    da = FrequencyTitle.objects.filter(term=term)
    if da.count() > 0:
        return da.first().idf
    else:
        return 0


def idf_detail(term):
    """Returns the IDF of a term based on description.
    """
    # da = BugReport.objects.count()
    # dt = FrequencyDetail.objects.filter(term=term).count()
    # return n.log2(float(da) / (1 + dt))
    da = FrequencyDetail.objects.filter(term=term)
    if da.count() > 0:
        return da.first().idf
    else:
        return 0


# The similarities

def similarity(text1, text2, idf_func=idf_title, tf_func=tf_title):
    """Returns the similarity between text1 and text2 based on the
    given IDF function.

    Parametersd
    ----------
    text1, text2 : string
        Strings to calculate similarity between
    idf_func = \{idf, idf_title, idf_detail\}
        The IDF function to use for calculating similarity
    """
    # common = set(tokenize(text1.title)).intersection(set(tokenize(text2.title)))
    # return n.sum([idf_func(w) for w in common])
    tf_vector1 = []
    idf_vector1 = []
    tf_idf_vector1 = []
    tf_vector2 = []
    idf_vector2 = []
    tf_idf_vector2 = []
    dot_product = []
    tf_idf_sum = 0.0
    magnitude1 = []
    magnitude2 = []
    magnitude1_sum = 0.0
    magnitude2_sum = 0.0
    cosine_similarity = 0.0
    union = set(tokenize(text1.title)).union(set(tokenize(text2.title)))
    for w in union:
        tf_vector1.append(tf_func(text1, w))
        idf_vector1.append(idf_func(w))
        tf_vector2.append(tf_func(text2, w))
        idf_vector2.append(idf_func(w))
    for i in range(len(union)):
        tf_idf_vector1.append(tf_vector1[i] * idf_vector1[i])
        tf_idf_vector2.append(tf_vector2[i] * idf_vector2[i])
        dot_product.append(tf_idf_vector1[i] * tf_idf_vector2[i])
        # magnitude1 += n.power(tf_idf_vector1[i], 2)
        magnitude1.append(n.power(tf_idf_vector1[i], 2))
        # magnitude2 += n.power(tf_idf_vector2[i], 2)
        magnitude2.append(n.power(tf_idf_vector2[i], 2))
        # tf_idf_sum += dot_product[i]
    tf_idf_sum = sum(dot_product)
    magnitude1_sum = sum(magnitude1)
    magnitude2_sum = sum(magnitude2)
    magnitude1_sum = n.sqrt(magnitude1_sum)
    magnitude2_sum = n.sqrt(magnitude2_sum)
    if magnitude1 != 0.0 and magnitude2 != 0.0:
        cosine_similarity = float(tf_idf_sum) / (magnitude1_sum * magnitude2_sum)
    # return str(cosine_similarity)
    # tf_idf_vector1 = [x for x in tf_idf_vector1 if x != 0]
    # tf_idf_vector2 = [x for x in tf_idf_vector2 if x != 0]
    # i1 = iter(tf_idf_vector1)
    # b1 = dict(zip(i1, i1))
    # i2 = iter(tf_idf_vector2)
    # b2 = dict(zip(i2, i2))
    # sim = gensim.matutils.cossim(b1, b2)
    return cosine_similarity
    # return sim


def feature_set(bug1, bug2):
    """Returns a list of (n-1) features for a pair of bug reports.
    """
    row = []
    for idf_func in [idf_title, idf_detail, idf]:
        for text1 in [bug1.title, bug1.detail_text(), bug1.all_text()]:
            for text2 in [bug2.title, bug2.detail_text(), bug2.all_text()]:
                row.append(similarity(text1, text2, idf_func))
    return row
