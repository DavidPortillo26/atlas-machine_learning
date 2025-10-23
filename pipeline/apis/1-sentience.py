#!/usr/bin/env python3
"""Fetch the home planets of all sentient species from SWAPI."""
import requests


SWAPI_SPECIES_URL = "https://swapi-api.alx-tools.com/api/species/"
SENTIENT_KEYWORD = "sentient"


def _is_sentient_species(species):
    """Return True when classification or designation mentions 'sentient'."""
    for key in ("classification", "designation"):
        value = species.get(key) or ""
        if SENTIENT_KEYWORD in value.lower():
            return True
    return False


def _get_planet_name(planet_url, cache):
    """Fetch and cache planet names by URL."""
    if not planet_url:
        return None

    if planet_url not in cache:
        response = requests.get(planet_url, timeout=10)
        response.raise_for_status()
        cache[planet_url] = response.json().get("name")

    return cache.get(planet_url)


def sentientPlanets():
    """
    Return the home planet names for every sentient species found in SWAPI.

    Sentience can be declared via the classification or designation field.
    Pagination is handled transparently for the species endpoint.
    """
    planets = []
    seen = set()
    cache = {}
    next_url = SWAPI_SPECIES_URL

    while next_url:
        response = requests.get(next_url, timeout=10)
        response.raise_for_status()

        data = response.json()
        for species in data.get("results", []):
            if not _is_sentient_species(species):
                continue

            planet_url = species.get("homeworld")
            planet_name = _get_planet_name(planet_url, cache)
            if planet_name and planet_name not in seen:
                seen.add(planet_name)
                planets.append(planet_name)

        next_url = data.get("next")

    return planets
