import math
import statistics
from datetime import timedelta

import rich.progress
from rich.progress import Progress as rProgress
from rich.traceback import install as niceTracebacks
from rich.console import Console
from rich.text import Text

########
niceTracebacks(show_locals=True)
########

rc = Console()


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


class ItemsPerSecondColumn(rich.progress.ProgressColumn):
    def render(self, task: "rich.progress.Task") -> Text:
        """Show time remaining."""
        if task.completed == 0:
            return Text("(0.0/s)", style="progress.elapsed")
        #
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("(0.0/s)", style="progress.elapsed")
        #
        itemsPS = roundToFirstSignificantDigits(task.completed / elapsed, 3, 3)
        return Text(f"({itemsPS}/s)", style="progress.elapsed")


class SecondsPerItemColumn(rich.progress.ProgressColumn):
    def render(self, task: "rich.progress.Task") -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("(0.0s/item)", style="progress.elapsed")
        #
        if task.completed == 0:
            return Text(f"({roundToFirstSignificantDigits(elapsed, 3, 3)}s/item)", style="progress.elapsed")
        #
        secPerItem = roundToFirstSignificantDigits(elapsed / task.completed, 3, 3)
        return Text(f"({secPerItem}s/item)", style="progress.elapsed")


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
        rich.progress.TimeRemainingColumn(),
        ItemsPerSecondColumn(),
        SecondsPerItemColumn(),
        console=console, transient=False)
