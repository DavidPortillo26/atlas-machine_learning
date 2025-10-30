#!/usr/bin/env python3
"""Insert a document into a MongoDB collection using kwargs."""


def insert_school(mongo_collection, **kwargs):
    """Insert a new document and return its generated _id."""
    if mongo_collection is None:
        return None
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
