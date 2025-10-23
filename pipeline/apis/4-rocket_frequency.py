#!/usr/bin/env python3
"""Display the number of launches per SpaceX rocket."""
import sys

import requests


LAUNCHES_URL = "https://api.spacexdata.com/v4/launches"
ROCKETS_URL = "https://api.spacexdata.com/v4/rockets"


def _fetch_json(url):
    """Retrieve JSON from the SpaceX API."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def _count_launches_by_rocket():
    """Return mapping of rocket id to number of launches."""
    counts = {}
    launches = _fetch_json(LAUNCHES_URL)
    for launch in launches:
        rocket_id = launch.get("rocket")
        if not rocket_id:
            continue
        counts[rocket_id] = counts.get(rocket_id, 0) + 1
    return counts


def _rocket_names():
    """Return mapping of rocket id to rocket name."""
    rockets = _fetch_json(ROCKETS_URL)
    return {rocket.get("id"): rocket.get("name") for rocket in rockets}


def display_launch_counts():
    """Print the launch counts per rocket."""
    counts = _count_launches_by_rocket()
    names = _rocket_names()

    results = []
    for rocket_id, count in counts.items():
        name = names.get(rocket_id, "Unknown rocket")
        results.append((name, count))

    results.sort(key=lambda item: (-item[1], item[0]))

    for name, count in results:
        print(f"{name}: {count}")


if __name__ == "__main__":
    try:
        display_launch_counts()
    except requests.RequestException as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
