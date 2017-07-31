import gensim
import logging

from django.core.management.base import BaseCommand

from gensim import corpora, similarities

from bugs.utils import tokenize
from bugs.models import BugReport


class Command(BaseCommand):
    help = "Learn from the current set of bug reports"

    def handle(self, *args, **options):

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        docs = []
        bugs = BugReport.objects.all()
        for bug in reversed(bugs):
            docs.append(tokenize(bug.text()))

        dictionary = corpora.Dictionary(docs)
        num_term = len(dictionary)
        corpus = [dictionary.doc2bow(text) for text in docs]
        tf_idf = gensim.models.TfidfModel(corpus)
        index = similarities.SparseMatrixSimilarity(tf_idf[corpus], num_features=num_term)

        dictionary.save('issues')
        tf_idf.save("tf-idf")
        index.save("index")


