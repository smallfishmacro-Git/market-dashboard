"""
Unit tests for market_time.session_is_complete.

Run with stdlib (no pytest required):
    python -m unittest discover -s tests
or under pytest:
    pytest tests/

`now` is fed in as a tz-aware UTC datetime so the tests pin a precise moment on
the wall clock regardless of the machine timezone — this is exactly the runner
condition (UTC clock) the helper exists to defend against.

------------------------------------------------------------------------------
RUNNER SIMULATION (manual, needs network + the real CSVs; not run here):

    TZ=UTC python fix_spx_ohlc.py && \
    TZ=UTC python data_updater.py && \
    python compute_market_risk_json.py && \
    python check_freshness.py

After a post-16:15-ET run the last thm date in market_risk.json must equal the
last completed NYSE session, and check_freshness.py must exit 0.
------------------------------------------------------------------------------
"""

import os
import sys
import unittest
from datetime import datetime, date, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_time import session_is_complete  # noqa: E401


def utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


class TestSessionIsComplete(unittest.TestCase):
    # 2025-01-15 is a Wednesday (EST, UTC-5). Close 16:00 ET == 21:00 UTC,
    # settle buffer pushes the deadline to 16:15 ET == 21:15 UTC.
    WED = date(2025, 1, 15)

    def test_midsession_utc_18_is_incomplete(self):
        # 18:00 UTC == 13:00 ET on a trading day -> session still open.
        self.assertFalse(session_is_complete(self.WED, now=utc(2025, 1, 15, 18)))

    def test_postclose_utc_23_is_complete(self):
        # 23:00 UTC == 18:00 ET same day -> session settled. This is the case
        # the old `== datetime.today()` UTC check got wrong.
        self.assertTrue(session_is_complete(self.WED, now=utc(2025, 1, 15, 23)))

    def test_weekend_friday_is_complete(self):
        # Friday's bar, evaluated on Saturday -> long settled.
        friday = date(2025, 1, 17)
        saturday_noon = utc(2025, 1, 18, 12)
        self.assertTrue(session_is_complete(friday, now=saturday_noon))

    def test_settle_buffer_boundary_est(self):
        # 21:14 UTC == 16:14 ET -> just before the 16:15 buffer -> incomplete.
        self.assertFalse(session_is_complete(self.WED, now=utc(2025, 1, 15, 21, 14)))
        # 21:15 UTC == 16:15 ET -> exactly at the buffer (>=) -> complete.
        self.assertTrue(session_is_complete(self.WED, now=utc(2025, 1, 15, 21, 15)))

    def test_settle_buffer_boundary_edt(self):
        # DST sanity: 2025-07-16 is a Wednesday in EDT (UTC-4). 16:15 ET == 20:15 UTC.
        wed_summer = date(2025, 7, 16)
        self.assertFalse(session_is_complete(wed_summer, now=utc(2025, 7, 16, 20, 14)))
        self.assertTrue(session_is_complete(wed_summer, now=utc(2025, 7, 16, 20, 15)))

    def test_future_bar_is_incomplete(self):
        # A bar dated tomorrow can never be complete now.
        self.assertFalse(session_is_complete(date(2025, 1, 16), now=utc(2025, 1, 15, 23)))

    def test_past_bar_always_complete(self):
        # Yesterday's bar is done even at pre-market this morning.
        self.assertTrue(session_is_complete(date(2025, 1, 14), now=utc(2025, 1, 15, 5)))

    def test_naive_now_treated_as_et(self):
        # A naive datetime is assumed to already be ET wall time.
        self.assertFalse(session_is_complete(self.WED, now=datetime(2025, 1, 15, 16, 14)))
        self.assertTrue(session_is_complete(self.WED, now=datetime(2025, 1, 15, 16, 15)))

    def test_accepts_datetime_and_timestamp_bar(self):
        # bar_date may arrive as a datetime/Timestamp (.date() is coerced).
        self.assertTrue(session_is_complete(datetime(2025, 1, 15, 0, 0),
                                            now=utc(2025, 1, 15, 23)))


if __name__ == "__main__":
    unittest.main()
