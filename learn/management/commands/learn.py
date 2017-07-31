from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Learn from the current set of bug reports"

    def handle(self, *args, **options):

        data = []  # This will finally end up as the training sample

        masters = BugReport.objects.filter(master=None)
        print("There are {} masters.".format(masters.count()))

        # Create positive training data
        for master in masters:
            for duplicate in master.duplicates.all():
                data.append(get_training_row(master, duplicate, 1))
            for bug1, bug2 in itertools.combinations(master.duplicates.all(), 2):
                data.append(get_training_row(bug1, bug2, 1))

        n = len(data)
        print("Generated {} positive samples.".format(n))

        # Create negative training data
        pairs = [pair for pair in itertools.combinations(masters, 2)]
        for bug1, bug2 in random.sample(pairs, n):
            data.append(get_training_row(bug1, bug2, 0))

        data = np.matrix(data)
        print("Total size of training data: {}".format(data.shape))
        np.savetxt('training.csv', data)

        model = LearningModel()
        model.learn(data)
        model.save()

        print("Successfully learned!")
