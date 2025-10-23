#!/usr/bin/env python3
"""Fetch starships from SWAPI that can carry a given number of passengers."""
import requests


SWAPI_STARSHIPS_URL = "https://swapi-api.alx-tools.com/api/starships/"


def _parse_passenger_capacity(value):
    """
    Convert the SWAPI passengers field to an integer.

    Non-numeric values (e.g. 'unknown', 'n/a') yield 0 so they never
    satisfy a positive passenger requirement.
    """
    if not value:
        return 0

    cleaned = value.replace(",", "").strip()
    if cleaned.isdigit():
        return int(cleaned)

    digits = "".join(ch for ch in cleaned if ch.isdigit())
    return int(digits) if digits else 0


def availableShips(passengerCount):
    """
    Return a list of starship names that can carry at
    least passengerCount passengers.

    Args:
        passengerCount (int): Minimum passenger capacity required.

    Returns:
        list[str]: Names of starships meeting the requirement.
    """
    if passengerCount <= 0:
        return []

    ships = []
    next_url = SWAPI_STARSHIPS_URL

    while next_url:
        response = requests.get(next_url, timeout=10)
        response.raise_for_status()

        data = response.json()
        for starship in data.get("results", []):
            capacity = _parse_passenger_capacity(starship.get("passengers", ""))
            if capacity >= passengerCount:
                ships.append(starship.get("name"))

        next_url = data.get("next")

    return ships
