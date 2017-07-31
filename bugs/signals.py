import numpy as n

from django.db.models.signals import post_save
from django.dispatch import receiver

from .utils import tokenize

from .models import BugReport


@receiver(post_save, sender=BugReport)
def bug_report_post_save(sender, instance, created, **kwargs):

