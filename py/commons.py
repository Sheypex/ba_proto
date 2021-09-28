import math
import statistics
from datetime import timedelta

import rich.progress
from rich.progress import Progress as rProgress
import rich.traceback
from rich.console import Console
from rich.text import Text
import rich.pretty

########
rc = Console()
rich.pretty.install(console=rc, indent_guides=True, max_length=2, expand_all=True)
rich.traceback.install(console=rc, show_locals=True)


########


# jamGeomean <- function
# (x,
#  na.rm=TRUE,
#  ...)
# {
#    ## Purpose is to calculate geometric mean while allowing for
#    ## positive and negative values
#    x2 <- mean(log2(1+abs(x))*sign(x));
#    sign(x2)*(2^abs(x2)-1);
# }
# taken from: https://jmw86069.github.io/splicejam/reference/jamGeomean.html
def jamGeomean(iterable):
    assert len(iterable) > 0
    step1 = [math.log(1 + abs(x), 2) * math.copysign(1, x) for x in iterable]
    m = statistics.mean(step1)
    return math.copysign(1, m) * ((2 ** abs(m)) - 1)


def list_compare(a, b):
    if type(a) != type(b):
        return False
    if type(a) != list:
        return a == b
    if len(a) != len(b):
        return False
    for a_, b_ in zip(a, b):
        if not list_compare(a_, b_):
            return False
    return True


def iround(num):
    return int(round(num, 0))


def roundToFirstSignificant(num, max=3):
    return roundToFirstSignificantDigits(num, 1, max)


def roundToFirstSignificantDigits(num, digits=1, max=3):
    assert digits >= 1
    assert max >= 0
    if round(num, max) == 0:
        return 0.0
    #
    firstSigDigit = 0
    while round(num, firstSigDigit) == 0.0:
        firstSigDigit += 1
    roundTo = firstSigDigit + digits - 1
    roundTo = max if roundTo > max else roundTo
    return round(num, roundTo)


def ema(x, mu=None, alpha=0.3):
    # taken from https://github.com/timwedde/rich-utils/blob/master/rich_utils/progress.py
    """
    Exponential moving average: smoothing to give progressively lower
    weights to older values.
    Parameters
    ----------
    x  : float
        New value to include in EMA.
    mu  : float, optional
        Previous EMA value.
    alpha  : float, optional
        Smoothing factor in range [0, 1], [default: 0.3].
        Increase to give more weight to recent values.
        Ranges from 0 (yields mu) to 1 (yields x).
    """
    return x if mu is None else (alpha * x) + (1 - alpha) * mu


class ItemsPerSecondColumn(rich.progress.ProgressColumn):
    max_refresh = 0.5

    def __init__(self):
        super().__init__()
        self.seen = dict()
        self.itemsPS = dict()

    def render(self, task: "rich.progress.Task") -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            self.seen[task.id] = 0
            self.itemsPS[task.id] = 0.0
            return Text("(0.0/s)", style="progress.elapsed")
        if task.finished:
            return Text(
                f"({roundToFirstSignificantDigits(task.completed / elapsed, 3, 3)}/s)",
                style="progress.elapsed",
            )
        if task.completed == 0:
            self.seen[task.id] = 0
            self.itemsPS[task.id] = 0.0
        if self.seen[task.id] < task.completed:
            self.itemsPS[task.id] = roundToFirstSignificantDigits(
                ema(
                    roundToFirstSignificantDigits(task.completed / elapsed, 3, 3),
                    self.itemsPS[task.id],
                ),
                3,
                3,
            )
            self.seen[task.id] = task.completed
        return Text(f"({self.itemsPS[task.id]}/s)", style="progress.elapsed")


class SecondsPerItemColumn(rich.progress.ProgressColumn):
    max_refresh = 0.5

    def __init__(self):
        super().__init__()
        self.seen = dict()
        self.secPerItem = dict()

    def render(self, task: "rich.progress.Task") -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            self.seen[task.id] = 0
            self.secPerItem[task.id] = 0.0
            return Text("(0.0s/item)", style="progress.elapsed")
        if task.finished:
            return Text(
                f"({roundToFirstSignificantDigits(elapsed / task.completed, 3, 3)}s/item)",
                style="progress.elapsed",
            )
        #
        if task.completed == 0:
            self.seen[task.id] = 0
            self.secPerItem[task.id] = roundToFirstSignificantDigits(elapsed, 3, 3)
            return Text(f"({self.secPerItem[task.id]}s/item)", style="progress.elapsed")
        #
        if self.seen[task.id] < task.completed:
            self.secPerItem[task.id] = roundToFirstSignificantDigits(
                ema(
                    roundToFirstSignificantDigits(elapsed / task.completed, 3, 3),
                    self.secPerItem[task.id],
                ),
                3,
                3,
            )
            self.seen[task.id] = task.completed
        return Text(f"({self.secPerItem[task.id]}s/item)", style="progress.elapsed")


# taken from https://github.com/timwedde/rich-utils/blob/master/rich_utils/progress.py
class SmartTimeRemainingColumn(rich.progress.ProgressColumn):
    max_refresh = 0.5

    def __init__(self, *args, **kwargs):
        self.seen = dict()
        self.avg_remaining_seconds = dict()
        self.smoothing = kwargs.get("smoothing", 0.3)
        del kwargs["smoothing"]
        super().__init__(*args, **kwargs)

    def render(self, task):
        remaining = task.time_remaining
        if remaining is None:
            self.seen[task.id] = 0
            self.avg_remaining_seconds[task.id] = 0.0
            return Text("-:--:--", style="progress.remaining")
        #
        if task.completed == 0:
            self.seen[task.id] = 0
            self.avg_remaining_seconds[task.id] = remaining
        #
        if self.seen[task.id] < task.completed:
            self.avg_remaining_seconds[task.id] = ema(
                remaining, self.avg_remaining_seconds[task.id], self.smoothing
            )
            self.seen[task.id] = task.completed
        return Text(
            str(timedelta(seconds=int(self.avg_remaining_seconds[task.id]))),
            style="progress.remaining",
        )


def stdProgress(console=None):
    if console is None:
        console = rc
    return rProgress(
        "[progress.description]{task.description}",
        rich.progress.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "({task.completed}/{task.total})",
        rich.progress.TimeElapsedColumn(),
        "eta:",
        SmartTimeRemainingColumn(),
        ItemsPerSecondColumn(),
        SecondsPerItemColumn(),
        console=console,
        transient=False,
    )
