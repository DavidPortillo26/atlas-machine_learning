#!/usr/bin/env python3
"""Display the earliest SpaceX launch with rocket and launchpad details."""
import sys

import requests


LAUNCHES_URL = "https://api.spacexdata.com/v5/launches"
ROCKETS_URL = "https://api.spacexdata.com/v4/rockets"
LAUNCHPADS_URL = "https://api.spacexdata.com/v4/launchpads"


def _fetch_json(url):
    """Retrieve JSON data from an API endpoint."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def _earliest_launch():
    """Return the earliest launch data structure from the SpaceX API."""
    launches = _fetch_json(LAUNCHES_URL)
    if not launches:
        return None
    # Enumerate to create stable order for ties on date_unix
    indexed = list(enumerate(launches))
    indexed.sort(key=lambda item: (item[1].get("date_unix", float("inf")), item[0]))
    return indexed[0][1]


def _rocket_name(rocket_id):
    """Fetch the rocket name given its id."""
    if not rocket_id:
        return "Unknown rocket"
    data = _fetch_json(f"{ROCKETS_URL}/{rocket_id}")
    return data.get("name", "Unknown rocket")


def _launchpad_details(launchpad_id):
    """Fetch launchpad name and locality by id."""
    if not launchpad_id:
        return "Unknown launchpad", "Unknown locality"
    data = _fetch_json(f"{LAUNCHPADS_URL}/{launchpad_id}")
    return data.get("name", "Unknown launchpad"), data.get("locality", "Unknown locality")


def display_first_launch():
    """Print the earliest SpaceX launch with formatted information."""
    launch = _earliest_launch()
    if not launch:
        return

    rocket = _rocket_name(launch.get("rocket"))
    launchpad_name, launchpad_locality = _launchpad_details(launch.get("launchpad"))

    name = launch.get("name", "Unknown")
    date_local = launch.get("date_local", "Unknown date")
    print(f"{name} ({date_local}) {rocket} - {launchpad_name} ({launchpad_locality})")


if __name__ == "__main__":
    try:
        display_first_launch()
    except requests.RequestException as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
