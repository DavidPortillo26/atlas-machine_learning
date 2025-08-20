#!/usr/bin/env python3

"""
Simple user intput loop script.

Prompts the user with Q: and responds with A:
Exits when the user inputs exit, quit, goodbye, or bye.
"""

EXIT_COMMANDS = ["exit", "quit", "goodbye", "bye"]

def main():
    while True:
        user_input = input("Q: ")
        if user_input.strip() in EXIT_COMMANDS:
            print("A: Goodbye!")
            break
        else:
            print(f"A:")

if __name__ == "__main__":
    main()
