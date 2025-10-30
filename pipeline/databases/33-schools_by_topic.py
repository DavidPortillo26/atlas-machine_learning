#!/usr/bin/env python3
"""Retrieve schools that cover a given topic."""


def schools_by_topic(mongo_collection, topic):
    """Return a list of schools where `topic` is in their topics array."""
    if mongo_collection is None:
        return []
    return list(mongo_collection.find({"topics": topic}))
