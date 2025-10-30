#!/usr/bin/env python3
"""Provide basic stats about Nginx logs stored in MongoDB."""

from pymongo import MongoClient


def main():
    """Compute and print the required log statistics."""
    client = MongoClient()
    collection = client.logs.nginx

    total = collection.count_documents({})
    print(f"{total} logs")

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods:")
    for method in methods:
        count = collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")

    status_checks = collection.count_documents({"method": "GET", "path": "/status"})
    print(f"{status_checks} status check")


if __name__ == "__main__":
    main()
