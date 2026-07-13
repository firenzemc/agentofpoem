from poemferry.rate_limit import RateLimiter


def test_allows_under_limit_then_blocks():
    rl = RateLimiter([(300, 3)], global_daily=1000)
    t = 1000.0
    assert rl.check("a", t) is None
    assert rl.check("a", t) is None
    assert rl.check("a", t) is None
    retry = rl.check("a", t)  # 4th hit in the 5-min window
    assert retry is not None and 0 < retry <= 300


def test_window_frees_after_expiry():
    rl = RateLimiter([(300, 1)], global_daily=1000)
    assert rl.check("a", 1000.0) is None
    assert rl.check("a", 1000.0) is not None
    assert rl.check("a", 1000.0 + 301) is None  # first hit aged out of the window


def test_per_ip_isolation():
    rl = RateLimiter([(300, 1)], global_daily=1000)
    assert rl.check("a", 1000.0) is None
    assert rl.check("b", 1000.0) is None  # a different IP has its own budget


def test_multiple_windows_all_enforced():
    # 2/5min but only 3/hour: the hour window bites on the 4th even across 5-min gaps
    rl = RateLimiter([(300, 2), (3600, 3)], global_daily=1000)
    assert rl.check("a", 0.0) is None
    assert rl.check("a", 0.0) is None
    assert rl.check("a", 400.0) is None  # new 5-min window, still under hourly
    assert rl.check("a", 800.0) is not None  # 4th in the hour → blocked


def test_global_daily_backstop():
    rl = RateLimiter([(300, 100)], global_daily=2)
    assert rl.check("a", 1000.0) is None
    assert rl.check("b", 1000.0) is None
    assert rl.check("c", 1000.0) is not None  # global cap hit regardless of IP
