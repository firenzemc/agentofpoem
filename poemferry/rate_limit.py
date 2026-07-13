"""In-memory per-IP rate limiting for the public profile.

Single-instance only (state lives in the process). Each client IP gets a sliding
window log; three windows (5min / hour / day) are checked against it. A global
daily counter is the runaway backstop. `check` records the hit when it allows,
and returns seconds-until-retry when it blocks. Internal profile skips this
entirely (limiter is None).
"""

from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, windows: list[tuple[int, int]], global_daily: int):
        # windows: [(seconds, max_hits), ...] — must be sorted by seconds ascending
        self.windows = sorted(windows)
        self.global_daily = global_daily
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.global_count = 0
        self.global_day = -1

    def check(self, ip: str, now: float) -> float | None:
        """None → allowed (and recorded). Otherwise seconds until a slot frees."""
        day = int(now // 86400)
        if day != self.global_day:
            self.global_day = day
            self.global_count = 0
        if self.global_count >= self.global_daily:
            return 86400 - (now % 86400)  # reset at the UTC day boundary

        dq = self.hits[ip]
        horizon = now - self.windows[-1][0]
        while dq and dq[0] < horizon:
            dq.popleft()
        if not dq:
            self.hits.pop(ip, None)  # keep the map from growing with idle IPs
            dq = self.hits[ip]

        for sec, mx in self.windows:
            cutoff = now - sec
            in_window = [t for t in dq if t >= cutoff]
            if len(in_window) >= mx:
                return sec - (now - in_window[0])  # oldest in-window entry ages out

        dq.append(now)
        self.global_count += 1
        return None
