#!/usr/bin/env python3
"""Print the location of a GitHub user via the GitHub REST API."""
import sys
from datetime import datetime, timezone

import requests


def _minutes_until(reset_timestamp):
    """Return minutes from now until the given Unix epoch timestamp."""
    reset_time = datetime.fromtimestamp(reset_timestamp, tz=timezone.utc)
    delta = reset_time - datetime.now(timezone.utc)
    return max(0, int(delta.total_seconds() // 60))


def _print_location(url):
    """Fetch and print the user location, handling special error responses."""
    response = requests.get(url, timeout=10)

    if response.status_code == 404:
        print("Not found")
        return

    if response.status_code == 403:
        reset = response.headers.get("X-RateLimit-Reset")
        try:
            minutes = _minutes_until(int(reset))
        except (TypeError, ValueError):
            minutes = 0
        print(f"Reset in {minutes} min")
        return

    response.raise_for_status()

    location = response.json().get("location")
    print(location or "Not found")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(0)
    _print_location(sys.argv[1])
