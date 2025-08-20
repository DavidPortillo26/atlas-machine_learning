#!/usr/bin/env python3
"""
Simple interactive Q/A loop script.

This script continuously prompts the user with 'Q:' to input a question or 
statement. It responds with 'A:' as a placeholder answer. The loop continues 
until the user types an exit command, at which point it prints 'A: Goodbye!' 
and ends.

Exit commands are: 'exit', 'quit', 'goodbye', or 'bye' (case-insensitive).
"""

# List of commands that will terminate the loop
EXIT_COMMANDS = ["exit", "quit", "goodbye", "bye"]

def main():
    """
    Main loop for interacting with the user.

    Continuously prompts the user for input. If the input is an exit command, 
    prints a goodbye message and exits. Otherwise, prints a placeholder 'A:'.
    """
    while True:
        # Prompt the user for input
        user_input = input("Q: ")

        # Check if the input is an exit command (case-insensitive)
        if user_input.strip().lower() in EXIT_COMMANDS:
            print("A: Goodbye!")
            break

        # Respond with a placeholder for other input
        else:
            print("A:")

# Ensure the main function runs only when the script is executed directly
if __name__ == "__main__":
    main()
