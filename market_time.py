"""
market_time.py
--------------
Exchange-clock helpers shared by the data pipeline.

The GitHub Actions runners are on UTC. Comparing a freshly-closed US trading
session against `datetime.today()` on a UTC clock deletes the same-day bar after
~21:00 UTC (the US close still equals UTC "today"), which is why post-close runs
used to publish BD-2 data until the next morning. Everything that decides whether
a daily bar is "done" must route through `session_is_complete()` so the decision
is made on the New-York clock, not the runner's.
"""

from datetime import datetime, date, time
from zoneinfo import ZoneInfo

# US equity exchange timezone (handles EST/EDT transitions automatically).
ET = ZoneInfo("America/New_York")

# Regular-session close is 16:00 ET. We wait a short settle buffer after the
# close before treating the bar as final, so a run firing seconds after 16:00
# doesn't grab a not-yet-settled print.
SETTLE_BUFFER_MINUTES = 15
SETTLE_DEADLINE = time(16, SETTLE_BUFFER_MINUTES)  # 16:15 ET


def now_et() -> datetime:
    """Current wall-clock time in US-Eastern (tz-aware)."""
    return datetime.now(ET)


def _as_date(d) -> date:
    """Coerce a date / datetime / pandas Timestamp to a plain ``date``."""
    # pandas Timestamp and stdlib datetime both subclass datetime; a plain
    # datetime.date does not, so this leaves real dates untouched.
    return d.date() if isinstance(d, datetime) else d


def session_is_complete(bar_date, now: datetime | None = None) -> bool:
    """Return True if the trading session labelled ``bar_date`` is fully settled.

    Decision is made on the New-York clock regardless of the runner timezone:
      * bar_date is in the past   -> True  (session long closed)
      * bar_date is in the future -> False (session hasn't happened)
      * bar_date is today (ET)    -> True only once we're at/after 16:15 ET

    ``now`` (tz-aware) may be supplied for testing; it is normalised to ET. A
    naive ``now`` is assumed to already be ET. Defaults to the live ET clock.
    """
    if now is None:
        now = now_et()
    elif now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    else:
        now = now.astimezone(ET)

    bar_date = _as_date(bar_date)
    today = now.date()

    if bar_date < today:
        return True
    if bar_date > today:
        return False
    return now.time() >= SETTLE_DEADLINE
