#!/usr/bin/env python3
"""Update topics for a school identified by name."""


def update_topics(mongo_collection, name, topics):
    """Replace the topics list for the matching school."""
    if mongo_collection is None:
        return
    mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}},
    )
