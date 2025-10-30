#!/usr/bin/env python3
"""List all documents in a MongoDB collection."""


def list_all(mongo_collection):
    """Return every document in the collection or an empty list."""
    if mongo_collection is None:
        return []
    return list(mongo_collection.find())
