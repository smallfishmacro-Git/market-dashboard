"""
barchart_browser.py
-------------------
Since 2026-08-28 www.barchart.com sits behind an AWS WAF JavaScript challenge
(CloudFront 202 + "gokuProps" page). Plain HTTP clients - requests or curl_cffi -
never execute that JS, so they never get the aws-waf-token / XSRF-TOKEN cookies
and /proxies/core-api answers 403 {"error":"Forbidden"}.

BrowserBarchartSession runs a real headless Chromium (Playwright):
  1. opens one Barchart page and waits until the challenge is solved and the
     XSRF-TOKEN cookie exists;
  2. issues every core-api call with fetch() from inside that page (same
     origin, same cookies, real browser fingerprint).

It is a drop-in for the interface data_updater.update_symbol() already uses:
    session.get(url, headers=..., params=..., timeout=...) -> .status_code/.headers/.text/.json()
    session.cookies.get_dict()
plus .backend / .failures (for data/diag/barchart_diag.json) and .close().
"""

import json
import time
from urllib.parse import unquote, urlencode

BARCHART_ORIGIN = "https://www.barchart.com"
WARMUP_URL = f"{BARCHART_ORIGIN}/stocks/quotes/%24SPX/price-history/historical"
TOKEN_COOKIE = "XSRF-TOKEN"
READY_COOKIE = "aws-waf-token"  # Barchart no longer sets XSRF-TOKEN; the WAF token is the readiness signal

_FETCH_JS = """
async ([url, xsrf, timeoutMs]) => {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
        const r = await fetch(url, {
            headers: {"accept": "application/json", "x-xsrf-token": xsrf},
            credentials: "same-origin",
            signal: ctrl.signal,
        });
        return {status: r.status, ctype: r.headers.get("content-type") || "", text: await r.text()};
    } finally {
        clearTimeout(t);
    }
}
"""


class _Resp:
    def __init__(self, status, headers, text):
        self.status_code = status
        self.headers = headers
        self.text = text

    def json(self):
        return json.loads(self.text)


class _Cookies:
    def __init__(self, ctx, origin):
        self._ctx = ctx
        self._origin = origin

    def get_dict(self):
        return {c["name"]: c["value"] for c in self._ctx.cookies(self._origin)}


class BrowserBarchartSession:
    backend = "playwright/chromium"

    def __init__(self, warmup_url=WARMUP_URL, origin=BARCHART_ORIGIN,
                 token_cookie=TOKEN_COOKIE, warmup_timeout_s=60):
        from playwright.sync_api import sync_playwright

        self.failures = []
        self._warmup_url = warmup_url
        self._token_cookie = token_cookie
        self._warmup_timeout_s = warmup_timeout_s
        self._rewarmed = False

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            channel="chromium",  # full Chromium, new headless mode (not headless-shell)
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        major = self._browser.version.split(".")[0]
        self._ctx = self._browser.new_context(
            user_agent=(f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        f"(KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36"),
            locale="en-US",
            timezone_id="America/New_York",
            viewport={"width": 1366, "height": 900},
        )
        self._ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        self.cookies = _Cookies(self._ctx, origin)
        self.page = self._ctx.new_page()
        self.ready = self._warmup()

    # ── helpers ────────────────────────────────────────────────────────────────
    def _record(self, entry):
        if len(self.failures) < 8:
            self.failures.append(entry)

    def _warmup(self):
        try:
            self.page.goto(self._warmup_url, wait_until="domcontentloaded",
                           timeout=self._warmup_timeout_s * 1000)
        except Exception as e:
            self._record({"stage": "warmup-goto", "error": f"{type(e).__name__}: {e}"[:300]})

        deadline = time.time() + self._warmup_timeout_s
        while time.time() < deadline:
            if READY_COOKIE in self.cookies.get_dict():
                return True
            self.page.wait_for_timeout(1000)

        try:
            body = self.page.content()[:400]
            title = self.page.title()
        except Exception as e:  # page may still be mid-navigation
            body, title = f"<unreadable: {type(e).__name__}>", None
        self._record({
            "stage": "warmup",
            "url": self.page.url,
            "title": title,
            "cookies_seen": sorted(self.cookies.get_dict().keys()),
            "body_head": body,
        })
        return False

    def _fetch(self, full_url, timeout):
        xsrf = unquote(unquote(self.cookies.get_dict().get(self._token_cookie, "")))
        res = self.page.evaluate(_FETCH_JS, [full_url, xsrf, int(timeout * 1000)])
        return _Resp(res["status"], {"content-type": res["ctype"]}, res["text"])

    # ── requests-compatible surface ───────────────────────────────────────────
    def get(self, url, headers=None, params=None, timeout=15, **kw):
        if "/proxies/" not in url:
            # data_updater only GETs the HTML page to obtain XSRF-TOKEN; the
            # browser warm-up already holds it, so skip the extra navigation.
            return _Resp(200, {"content-type": "text/html"}, "")

        full_url = url + ("?" + urlencode(params) if params else "")
        symbol = (params or {}).get("symbol")
        try:
            r = self._fetch(full_url, timeout)
            if r.status_code in (202, 403, 405) and not self._rewarmed:
                # WAF token expired mid-run: solve the challenge once more, retry.
                self._rewarmed = True
                self._warmup()
                r = self._fetch(full_url, timeout)
        except Exception as e:
            self._record({"url": url, "symbol": symbol,
                          "error": f"{type(e).__name__}: {e}"[:300]})
            raise

        ctype = (r.headers.get("content-type") or "").lower()
        if r.status_code != 200 or "json" not in ctype:
            self._record({
                "url": url,
                "symbol": symbol,
                "status": r.status_code,
                "content_type": ctype,
                "cookies_seen": sorted(self.cookies.get_dict().keys()),
                "body_head": r.text[:400],
            })
        return r

    def close(self):
        for fn in (self._browser.close, self._pw.stop):
            try:
                fn()
            except Exception:
                pass
