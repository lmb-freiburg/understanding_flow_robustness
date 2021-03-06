import sys

import progressbar
from blessings import Terminal


class TermLogger:
    def __init__(self, n_epochs, train_size, valid_size, attack_size):
        self.n_epochs = n_epochs
        self.train_size = train_size
        self.valid_size = valid_size
        self.attack_size = attack_size
        self.t = Terminal()
        s = 13
        e = 1  # epoch bar position
        tr = 3  # train bar position
        ts = 6  # valid bar position
        ta = 9  # attack bar position
        h = self.t.height
        if h is None:
            h = 25

        for _ in range(10):
            print("")
        self.epoch_bar = progressbar.ProgressBar(
            maxval=n_epochs, fd=Writer(self.t, (0, h - s + e))
        )

        self.train_writer = Writer(self.t, (0, h - s + tr))
        self.train_bar_writer = Writer(self.t, (0, h - s + tr + 1))

        self.valid_writer = Writer(self.t, (0, h - s + ts))
        self.valid_bar_writer = Writer(self.t, (0, h - s + ts + 1))

        self.attack_writer = Writer(self.t, (0, h - s + ta))
        self.attack_bar_writer = Writer(self.t, (0, h - s + ta + 1))

        self.reset_train_bar()
        self.reset_valid_bar()
        self.reset_attack_bar()

    def reset_train_bar(self):
        self.train_bar = progressbar.ProgressBar(
            maxval=self.train_size, fd=self.train_bar_writer
        ).start()

    def reset_valid_bar(self):
        self.valid_bar = progressbar.ProgressBar(
            maxval=self.valid_size, fd=self.valid_bar_writer
        ).start()

    def reset_attack_bar(self):
        self.attack_bar = progressbar.ProgressBar(
            maxval=self.attack_size, fd=self.attack_bar_writer
        ).start()


class Writer:
    """Create an object with a write method that writes to a
    specific place on the screen, defined at instantiation.

    This is the glue between blessings and progressbar.
    """

    def __init__(self, t, location):
        """
        Input: location - tuple of ints (x, y), the position
                        of the bar in the terminal
        """
        self.location = location
        self.t = t

    def write(self, string):
        with self.t.location(*self.location):
            sys.stdout.write("\033[K")
            print(string)

    @staticmethod
    def flush():
        return


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.min = [10e6] * i
        self.max = [0] * i
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert len(val) == self.meters
        self.count += n
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count
            self.min[i] = v if v < self.min[i] else self.min[i]
            self.max[i] = v if v > self.max[i] else self.max[i]

    def __repr__(self):
        val = " ".join(["{:.{}f}".format(v, self.precision) for v in self.val])
        avg = " ".join(["{:.{}f}".format(a, self.precision) for a in self.avg])
        return f"{val} ({avg})"
