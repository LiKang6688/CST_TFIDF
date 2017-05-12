import numpy as n

from django.db.models.signals import post_save
from django.dispatch import receiver

from info.models import Frequency, FrequencyDetail, FrequencyTitle
from info.utils import tokenize

from .models import BugReport


@receiver(post_save, sender=BugReport)
def bug_report_post_save(sender, instance, created, **kwargs):
    """Updates the frequencies based on updates in the bug report.

    Parameters
    ----------
    sender : class
        The sender class.
    instance : object
        The object of `sender` that caused the signal to be sent.
    created : bool
        Whether instance is created (true) or updated (false).
    """

    # Update the FrequencyTitle table
    terms = tokenize(instance.title)
    for t in set(terms):
        values = {'freq': terms.count(t)/float(len(set(terms))),
                  'idf': n.log2(
                       float(BugReport.objects.count()) / (1 + FrequencyTitle.objects.filter(term=t).count()))
                  }
        FrequencyTitle.objects.update_or_create(bug=instance, term=t, defaults=values)

    # Update the FrequencyDetail table
    terms = tokenize(instance.detail_text())
    for t in set(terms):
        values = {'freq': terms.count(t)/float(len(set(terms))),
                  'idf': n.log2(
                      float(BugReport.objects.count()) / (1 + FrequencyDetail.objects.filter(term=t).count()))
                  }
        FrequencyDetail.objects.update_or_create(bug=instance, term=t, defaults=values)

    # Update the Frequency table
    terms = tokenize(instance.all_text())
    for t in set(terms):
        values = {'freq': terms.count(t)/float(len(set(terms))),
                  'idf': n.log2(
                      float(BugReport.objects.count()) / (1 + Frequency.objects.filter(term=t).count()))
                  }
        Frequency.objects.update_or_create(bug=instance, term=t, defaults=values)
